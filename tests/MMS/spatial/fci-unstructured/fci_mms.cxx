#include "bout/bout.hxx"
#include "bout/build_config.hxx"
#include "bout/difops.hxx"
#include "bout/field.hxx"
#include "bout/field3d.hxx"
#include "bout/field_factory.hxx"
#include "bout/fv_ops.hxx"
#include "bout/globals.hxx"
#include "bout/options.hxx"
#include "bout/options_io.hxx"
#include "bout/petsc_operators.hxx"
#include "bout/petsclib.hxx"

#include <petscksp.h>

#include "bout/utils.hxx"
#include <array>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {
auto fci_op_test(const std::string& name, Options& dump, const Field3D& input,
                 const Field3D& result) {
  auto* mesh = input.getMesh();
  const Field3D solution{FieldFactory::get()->create3D(fmt::format("{}_solution", name),
                                                       Options::getRoot(), mesh)};
  const Field3D error{result - solution};

  dump[fmt::format("{}_l_2", name)] = sqrt(mean(SQ(error), true, "RGN_NOBNDRY"));
  dump[fmt::format("{}_l_inf", name)] = max(abs(error), true, "RGN_NOBNDRY");

  dump[fmt::format("{}_result", name)] = result;
  dump[fmt::format("{}_error", name)] = error;
  dump[fmt::format("{}_input", name)] = input;
  dump[fmt::format("{}_solution", name)] = solution;

  for (int slice = 1; slice < mesh->ystart; ++slice) {
    dump[fmt::format("{}_input.ynext(-{})", name, slice)] = input.ynext(-slice);
    dump[fmt::format("{}_input.ynext({})", name, slice)] = input.ynext(slice);
  }
}

/// Solution error of a steady diffusion problem with the parallel conduction
/// operator L (K = 1): (I - c L) f = S, S manufactured from the exact f. As L
/// is linear, the error delta = f_h - f solves (I - c L) delta = c e, where
/// e = L f - L_exact f is the operator error. Solved with GMRES on the
/// interior cells (matrix-free, the production operator in every product).
struct SolveContext {
  Mesh* mesh;
  bout::ConductionMethod method;
  BoutReal c;
  Field3D K;
};

void unpack(const PetscScalar* x, Field3D& f) {
  int i = 0;
  BOUT_FOR_SERIAL(ind, f.getRegion("RGN_NOBNDRY")) { f[ind] = x[i++]; }
}

void pack(const Field3D& f, PetscScalar* x) {
  int i = 0;
  BOUT_FOR_SERIAL(ind, f.getRegion("RGN_NOBNDRY")) { x[i++] = f[ind]; }
}

PetscErrorCode applyShell(Mat A, Vec x, Vec y) {
  SolveContext* ctx{nullptr};
  PetscCall(MatShellGetContext(A, &ctx));
  Field3D f{0.0, ctx->mesh};
  const PetscScalar* xa{nullptr};
  PetscCall(VecGetArrayRead(x, &xa));
  unpack(xa, f);
  PetscCall(VecRestoreArrayRead(x, &xa));
  ctx->mesh->communicate(f);
  Field3D flow;
  const Field3D Lf = Div_par_K_Grad_par_mod(ctx->K, f, flow, false, ctx->method);
  const Field3D result = f - ctx->c * Lf;
  PetscScalar* ya{nullptr};
  PetscCall(VecGetArray(y, &ya));
  pack(result, ya);
  PetscCall(VecRestoreArray(y, &ya));
  return PETSC_SUCCESS;
}

/// Returns {rms(delta), GMRES iterations, converged (1/0)}
std::array<BoutReal, 3> solutionError(Mesh* mesh, bout::ConductionMethod method,
                                      BoutReal c, const Field3D& operator_error) {
  const PetscLib lib;
  Field3D K{1.0, mesh};
  mesh->communicate(K);
  SolveContext ctx{mesh, method, c, K};
  const auto n_local = static_cast<PetscInt>(mesh->getRegion3D("RGN_NOBNDRY").size());

  Vec b{nullptr};
  Vec delta{nullptr};
  BOUT_DO_PETSC(VecCreateMPI(BoutComm::get(), n_local, PETSC_DETERMINE, &b));
  BOUT_DO_PETSC(VecDuplicate(b, &delta));
  PetscScalar* ba{nullptr};
  BOUT_DO_PETSC(VecGetArray(b, &ba));
  pack(c * operator_error, ba);
  BOUT_DO_PETSC(VecRestoreArray(b, &ba));

  Mat A{nullptr};
  BOUT_DO_PETSC(MatCreateShell(BoutComm::get(), n_local, n_local, PETSC_DETERMINE,
                               PETSC_DETERMINE, &ctx, &A));
  BOUT_DO_PETSC(
      MatShellSetOperation(A, MATOP_MULT, reinterpret_cast<void (*)(void)>(applyShell)));
  KSP ksp{nullptr};
  BOUT_DO_PETSC(KSPCreate(BoutComm::get(), &ksp));
  BOUT_DO_PETSC(KSPSetOperators(ksp, A, A));
  BOUT_DO_PETSC(KSPSetType(ksp, KSPGMRES));
  BOUT_DO_PETSC(KSPGMRESSetRestart(ksp, 300));
  PC pc{nullptr};
  BOUT_DO_PETSC(KSPGetPC(ksp, &pc));
  BOUT_DO_PETSC(PCSetType(pc, PCNONE));
  const BoutReal rtol = Options::root()["solve_rtol"]
                            .doc("Relative tolerance of the GMRES solves")
                            .withDefault(1e-10);
  BOUT_DO_PETSC(KSPSetTolerances(ksp, rtol, 1e-50, 1e10, 20000));
  BOUT_DO_PETSC(KSPSolve(ksp, b, delta));
  PetscInt its{0};
  KSPConvergedReason reason{};
  BOUT_DO_PETSC(KSPGetIterationNumber(ksp, &its));
  BOUT_DO_PETSC(KSPGetConvergedReason(ksp, &reason));

  Field3D d{0.0, mesh};
  const PetscScalar* da{nullptr};
  BOUT_DO_PETSC(VecGetArrayRead(delta, &da));
  unpack(da, d);
  BOUT_DO_PETSC(VecRestoreArrayRead(delta, &da));
  const BoutReal rms = sqrt(mean(SQ(d), true, "RGN_NOBNDRY"));

  BOUT_DO_PETSC(KSPDestroy(&ksp));
  BOUT_DO_PETSC(MatDestroy(&A));
  BOUT_DO_PETSC(VecDestroy(&b));
  BOUT_DO_PETSC(VecDestroy(&delta));
  return {rms, static_cast<BoutReal>(its), reason > 0 ? 1.0 : 0.0};
}

/// Energy test of a conduction method (K = 1, no boundary flux): integrate
/// du/dt = L u from a random initial state to time @p end_time with the
/// classical Runge-Kutta method (explicit, so no SUNDIALS or CVODE), step
/// 2 / rho with rho the spectral radius of L from a power iteration.
/// Returns {max_t E(t) / E(0), E(end) / E(0), steps}, E = sum dV u^2.
std::array<BoutReal, 3> energyGrowth(Mesh* mesh, bout::ConductionMethod method,
                                     BoutReal end_time) {
  Field3D one{1.0, mesh};
  mesh->communicate(one);
  Field3D flow;
  auto L = [&](const Field3D& v) {
    Field3D w = copy(v);
    mesh->communicate(w);
    return Div_par_K_Grad_par_mod(one, w, flow, false, method);
  };
  Field3D volume{mesh};
  mesh->get(volume, "dagp_us_volume", 0.0, false);
  volume *= mesh->getCoordinates()->dy();
  auto energy = [&](const Field3D& v) {
    return mean(volume * SQ(v), true, "RGN_NOBNDRY");
  };

  // Random initial state (seeded per rank: reproducible for a given
  // decomposition)
  std::mt19937 generator(12345U + static_cast<unsigned>(BoutComm::rank()));
  std::uniform_real_distribution<BoutReal> uniform(-1.0, 1.0);
  Field3D u{0.0, mesh};
  BOUT_FOR_SERIAL(i, u.getRegion("RGN_NOBNDRY")) { u[i] = uniform(generator); }

  // Spectral radius
  Field3D v = copy(u);
  BoutReal rho = 0.0;
  for (int it = 0; it < 100; ++it) {
    const BoutReal norm = sqrt(mean(SQ(v), true, "RGN_NOBNDRY"));
    v = L(v / norm);
    rho = sqrt(mean(SQ(v), true, "RGN_NOBNDRY"));
  }
  const int steps = static_cast<int>(std::ceil(end_time * rho / 2.0));
  const BoutReal dt = end_time / steps;

  const BoutReal energy0 = energy(u);
  BoutReal max_ratio = 1.0;
  for (int step = 0; step < steps; ++step) {
    const Field3D k1 = L(u);
    const Field3D k2 = L(u + 0.5 * dt * k1);
    const Field3D k3 = L(u + 0.5 * dt * k2);
    const Field3D k4 = L(u + dt * k3);
    u += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    max_ratio = std::max(max_ratio, energy(u) / energy0);
  }
  return {max_ratio, energy(u) / energy0, static_cast<BoutReal>(steps)};
}
} // namespace

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);

  using bout::globals::mesh;

  Field3D input{FieldFactory::get()->create3D("input_field", Options::getRoot(), mesh)};
  Field3D K{FieldFactory::get()->create3D("K", Options::getRoot(), mesh)};

  // Communicate to calculate parallel transform.
  mesh->communicate(input, K);

  Options dump;
  // Add mesh geometry variables
  mesh->outputVars(dump);

  // Dummy variable for *_mod overloads
  Field3D flow_ylow;

  fci_op_test("grad_par", dump, input, Grad_par(input));
  fci_op_test("grad2_par2", dump, input, Grad2_par2(input));
  fci_op_test("div_par", dump, input, Div_par(input));
  fci_op_test("div_par_K_grad_par", dump, input, Div_par_K_Grad_par(K, input));
  fci_op_test("div_par_K_grad_par_mod", dump, input,
              Div_par_K_Grad_par_mod(K, input, flow_ylow));
  // Support-operator method (matrices from the grid file)
  const Field3D som = Div_par_K_Grad_par_mod(K, input, flow_ylow, false,
                                             bout::ConductionMethod::PetscOperator);
  fci_op_test("div_par_K_grad_par_som", dump, input, som);
  {
    // Conservation in the physical volume dy * dagp_us_volume (field lines
    // are closed: no parallel boundaries)
    Field3D volume{mesh};
    mesh->get(volume, "dagp_us_volume", 0.0, false);
    volume *= mesh->getCoordinates()->dy();
    dump["div_par_K_grad_par_som_integral"] =
        mean(volume * som, true, "RGN_NOBNDRY")
        / mean(volume * abs(som), true, "RGN_NOBNDRY");
  }
  // K2 divergence (matrices from the grid file; only if requested)
  const bool k2 = Options::root()["k2"]
                      .doc("Test the K2 divergence (needs the K2 matrices in the grid)")
                      .withDefault(false);
  if (k2) {
    const Field3D k2_result = Div_par_K_Grad_par_mod(
        K, input, flow_ylow, false, bout::ConductionMethod::PetscOperatorK2);
    fci_op_test("div_par_K_grad_par_k2", dump, input, k2_result);
    Field3D volume{mesh};
    mesh->get(volume, "dagp_us_volume", 0.0, false);
    volume *= mesh->getCoordinates()->dy();
    dump["div_par_K_grad_par_k2_integral"] =
        mean(volume * k2_result, true, "RGN_NOBNDRY")
        / mean(volume * abs(k2_result), true, "RGN_NOBNDRY");
  }
  fci_op_test("laplace_par", dump, input, Laplace_par(input));

  // Finite volume methods
  fci_op_test("FV_div_par_mod", dump, input, FV::Div_par_mod(input, K, K, flow_ylow));
  fci_op_test("FV_div_par_fvv", dump, input, FV::Div_par_fvv(input, K, K));

  // Support operators (PetscOperators::Parallel, matrices from the grid file)
  if (Options::root()["support_operators"]
          .doc("Test the support operators (needs the operator data in the grid)")
          .withDefault(true)) {
#if BOUT_HAS_PETSC
    const auto par = PetscOperators::Cache::getParallel(mesh);
    fci_op_test("so_grad_par", dump, input, par->Grad_par(input));
    fci_op_test("so_div_par", dump, input, par->Div_par(input));
    fci_op_test("so_laplace_par", dump, input, par->Div_par_Grad_par(input));
    fci_op_test("so_div_par_K_grad_par", dump, input,
                par->Div_par_K_Grad_par(K, input, false));
#else
    throw BoutException("The support operators need PETSc");
#endif
  }

  // Solution error of (I - c L) f = S for both conduction methods (K = 1)
  for (const auto& c_string : strsplit(
           Options::root()["solve_c"]
               .doc("Comma-separated diffusion strengths c of the solution-error test")
               .withDefault<std::string>(""),
           ',')) {
    if (trim(c_string).empty()) {
      continue;
    }
    const BoutReal c = stringToReal(trim(c_string));
    const Field3D exact{
        FieldFactory::get()->create3D("laplace_par_solution", Options::getRoot(), mesh)};
    Field3D one{1.0, mesh};
    mesh->communicate(one);
    const std::map<std::string, bout::ConductionMethod> known{
        {"orig", bout::ConductionMethod::Original},
        {"som", bout::ConductionMethod::PetscOperator},
        {"k2", bout::ConductionMethod::PetscOperatorK2}};
    std::vector<std::pair<std::string, bout::ConductionMethod>> methods;
    for (const auto& name :
         strsplit(Options::root()["solve_methods"]
                      .doc("Comma-separated conduction methods of the solution-error "
                           "test: orig, som, k2")
                      .withDefault<std::string>("orig,som"),
                  ',')) {
      const auto search = known.find(trim(name));
      if (search == known.end()) {
        throw BoutException("Unknown solve method '{}'", name);
      }
      methods.emplace_back(search->first, search->second);
    }
    for (const auto& [label, method] : methods) {
      const Field3D e =
          Div_par_K_Grad_par_mod(one, input, flow_ylow, false, method) - exact;
      const auto [rms, its, ok] = solutionError(mesh, method, c, e);
      const std::string key = fmt::format("solve_{}_c{:g}", label, c);
      dump[key + "_error"] = rms;
      dump[key + "_iterations"] = its;
      dump[key + "_converged"] = ok;
    }
  }

  // Energy test of the K2 conduction (random initial state, K = 1)
  const auto energy_time = Options::root()["energy_time"]
                               .doc("End time of the K2 energy test (0: off)")
                               .withDefault(0.0);
  if (energy_time > 0.0) {
    const auto [max_ratio, final_ratio, steps] =
        energyGrowth(mesh, bout::ConductionMethod::PetscOperatorK2, energy_time);
    dump["energy_k2_max_ratio"] = max_ratio;
    dump["energy_k2_final_ratio"] = final_ratio;
    dump["energy_k2_steps"] = steps;
  }

  {
    // Perpendicular diffusion (poloidal-plane form on unstructured meshes).
    // In its own scope so that the PETSc objects are released before
    // BoutFinalise
    const auto dagp = FV::Div_a_Grad_perp_FCI::create(mesh, 1.0);
    Field3D flow_R;
    Field3D flow_Z;
    Field3D wall_outflow;
    const Field3D dagp_result = (*dagp)(K, input, flow_R, flow_Z, wall_outflow, false);
    fci_op_test("FV_div_a_grad_perp", dump, input, dagp_result);
    fci_op_test("FV_div_a_grad_perp_flowZ", dump, input, flow_Z);
    fci_op_test("FV_div_a_grad_perp_upwind", dump, input, (*dagp)(K, input, true));

    // Conservation: every plane is closed (periodic in Z, zero-flux walls in
    // R), so the volume integral vanishes
    Field3D volume{mesh};
    mesh->get(volume, "dagp_us_volume", 0.0, false);
    dump["FV_div_a_grad_perp_integral"] =
        mean(volume * dagp_result, true, "RGN_NOBNDRY")
        / mean(volume * abs(dagp_result), true, "RGN_NOBNDRY");
    // With Dirichlet wall nodes the flow into the walls closes the balance
    // (and is zero on closed planes)
    dump["FV_div_a_grad_perp_wall_integral"] =
        mean(volume * (dagp_result + wall_outflow), true, "RGN_NOBNDRY")
        / mean(volume * abs(dagp_result), true, "RGN_NOBNDRY");
    dump["FV_div_a_grad_perp_wall_flow"] =
        mean(volume * abs(wall_outflow), true, "RGN_NOBNDRY")
        / mean(volume * abs(dagp_result), true, "RGN_NOBNDRY");
  }

  bout::writeDefaultOutputFile(dump);

  BoutFinalise();
}
