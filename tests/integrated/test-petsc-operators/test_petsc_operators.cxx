#include "bout/bout.hxx"
#include "bout/difops.hxx"
#include "bout/field.hxx"
#include "bout/field3d.hxx"
#include "bout/field_factory.hxx"
#include "bout/globals.hxx"
#include "bout/options.hxx"
#include "bout/options_io.hxx"
#include "bout/petsc_operators.hxx"
#include "bout/region.hxx"

#include <chrono>

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);

  {
    const PetscOperators ops(bout::globals::mesh);

    // Parallel operators
    const auto parallel = ops.getParallel();

    // Test field
    Field3D f = FieldFactory::get()->create3D("sin(y) + cos(z)");
    bout::globals::mesh->communicate(f);
    f.applyParallelBoundary("parallel_dirichlet_o2");

    Field3D f_neumann = FieldFactory::get()->create3D("sin(y) + cos(z)");
    bout::globals::mesh->communicate(f_neumann);
    f_neumann.applyParallelBoundary("parallel_neumann_o1");

    auto forward_op = parallel.Restrict_minus * parallel.Forward;

    Options dump;
    dump["f"] = f;
    dump["dV"] = parallel.dV;

    // Test1 : (parallel.Restrict_minus * parallel.Inject_plus) is the identity in the interior (not X boundaries)

    dump["grad_par_op"] = parallel.Grad_par(f);
    Field3D forward{0.0};
    BOUT_FOR(i, forward.getRegion("RGN_NOBNDRY")) { forward[i] = f.yup()[i.yp()]; }

    dump["forward"] = forward;

    dump["forward_op"] = forward_op(f);

    dump["grad_par_yud"] = Grad_par(f);

    // Test2 : Sum of dV * Div_par over domain is zero

    dump["div_par_op"] = parallel.Div_par(f);

    auto* coords = bout::globals::mesh->getCoordinates();
    dump["div_par_yud"] = Div_par(f);

    dump["div_par_grad_par_op"] = parallel.Div_par_Grad_par(f_neumann);
    dump["div_par_grad_par_yud"] = Div_par_K_Grad_par(1.0, f_neumann);

    // Test3 : the cache keeps the parallel operators alive between calls, so
    // the second request (with no holder in between) reuses them
    using steady = std::chrono::steady_clock;
    const PetscOperators::Parallel* first{nullptr};
    double time_first{0.0};
    {
      const auto start = steady::now();
      const auto par = PetscOperators::Cache::getParallel(bout::globals::mesh);
      time_first = std::chrono::duration<double>(steady::now() - start).count();
      first = par.get();
    }
    const auto start = steady::now();
    const auto par = PetscOperators::Cache::getParallel(bout::globals::mesh);
    const double time_second =
        std::chrono::duration<double>(steady::now() - start).count();
    dump["cache_reused"] = (par.get() == first) ? 1 : 0;
    dump["cache_time_first"] = time_first;
    dump["cache_time_second"] = time_second;

    // Test4 : support-operator conduction with a variable coefficient is
    // conservative (sum of dV * divergence vanishes)
    Field3D K = FieldFactory::get()->create3D("1 + 0.5 * sin(y) * cos(z)");
    bout::globals::mesh->communicate(K);
    K.applyParallelBoundary("parallel_neumann_o1");
    Field3D flow;
    dump["div_par_k_grad_par_som"] = Div_par_K_Grad_par_mod(
        K, f_neumann, flow, false, bout::ConductionMethod::PetscOperator);
    dump["div_par_k_grad_par_orig"] = Div_par_K_Grad_par_mod(
        K, f_neumann, flow, false, bout::ConductionMethod::Original);

    bout::writeDefaultOutputFile(dump);
  }

  BoutFinalise();
  return 0;
}
