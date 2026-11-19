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

#include <cmath>
#include <string>

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
  BOUT_DO_PETSC(KSPSetTolerances(ksp, 1e-10, 1e-50, 1e10, 20000));
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
    for (const auto& [label, method] :
         {std::pair{"orig", bout::ConductionMethod::Original},
          std::pair{"som", bout::ConductionMethod::PetscOperator}}) {
      const Field3D e =
          Div_par_K_Grad_par_mod(one, input, flow_ylow, false, method) - exact;
      const auto [rms, its, ok] = solutionError(mesh, method, c, e);
      const std::string key = fmt::format("solve_{}_c{:g}", label, c);
      dump[key + "_error"] = rms;
      dump[key + "_iterations"] = its;
      dump[key + "_converged"] = ok;
    }
  }

  {
    // Perpendicular diffusion (poloidal-plane form on unstructured meshes).
    // In its own scope so that the PETSc objects are released before
    // BoutFinalise
    const auto dagp = FV::Div_a_Grad_perp_FCI::create(mesh, 1.0);
    Field3D flow_R;
    Field3D flow_Z;
    const Field3D dagp_result = (*dagp)(K, input, flow_R, flow_Z, false);
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
  }

  bout::writeDefaultOutputFile(dump);

  BoutFinalise();
}
