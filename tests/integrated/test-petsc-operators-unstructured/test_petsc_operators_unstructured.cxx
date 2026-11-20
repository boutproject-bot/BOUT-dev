// Support operators on an unstructured grid with parallel (y) walls:
// cell volume and boundary fluxes. See runtest.

#include "bout/boundary_factory.hxx"
#include "bout/boundary_region_iter.hxx"
#include "bout/bout.hxx"
#include "bout/boutexception.hxx"
#include "bout/coordinates.hxx"
#include "bout/difops.hxx"
#include "bout/field3d.hxx"
#include "bout/field_factory.hxx"
#include "bout/globals.hxx"
#include "bout/options.hxx"
#include "bout/options_io.hxx"
#include "bout/output.hxx"
#include "bout/parallel_boundary_op.hxx"
#include "bout/petsc_operators.hxx"

#include <memory>
#include <string>

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);

  {
    using bout::globals::mesh;
    // Optional length normalisation of the metric, as Hermes-3 does on FCI
    // grids (g_ij / L^2, J / L^3): the operators must then use the
    // normalised lengths, also the K2 matrices whose weights were computed
    // from the (unnormalised) metric in the grid file
    const BoutReal length_normalisation =
        Options::root()["length_normalisation"].withDefault(1.0);
    if (length_normalisation != 1.0) {
      MetricNormaliser norm;
      norm.g = length_normalisation * length_normalisation;
      norm.J = length_normalisation * length_normalisation * length_normalisation;
      mesh->getCoordinates()->normaliseMetric(norm);
    }
    const auto par = PetscOperators::Cache::getParallel(mesh);

    // f linear along the (straight) field lines, with its values at the
    // walls (half a cell beyond the last cells) as boundary values, set in
    // three ways that must agree:
    //  - fb: field values (f at the nearest wall in each boundary cell) through
    //    BoundaryFactory + clone for every parallel region;
    //  - f_label: the same field values through the label-based
    //    Field3D::applyParallelBoundary (labels are not unique);
    //  - f_expr: the expression form, evaluated at the boundary point.
    Field3D f = FieldFactory::get()->create3D("f", Options::getRoot(), mesh);
    Field3D fb = FieldFactory::get()->create3D("f_wall", Options::getRoot(), mesh);
    mesh->communicate(f);
    Field3D f_label = copy(f);
    Field3D f_expr = copy(f);
    auto* factory = BoundaryFactory::getInstance();
    for (const auto& region : mesh->getBoundariesPar()) {
      const std::unique_ptr<BoundaryOpPar> tmp{
          dynamic_cast<BoundaryOpPar*>(factory->create("parallel_dirichlet_o2", region))};
      const std::unique_ptr<BoundaryOpPar> op{tmp->clone(region.get(), &fb)};
      op->apply(f);
    }
    for (const std::string label : {"FCI_forward", "FCI_backward"}) {
      f_label.applyParallelBoundary(label, "parallel_dirichlet_o2", &fb);
    }
    f_expr.applyParallelBoundary("parallel_dirichlet_o2(10 * y / (2 * pi))");

    Field3D K{1.0, mesh};
    mesh->communicate(K);
    K.applyParallelBoundary("parallel_neumann_o1");

    Field3D volume{mesh};
    mesh->get(volume, "dagp_us_volume", 0.0, false);

    Options dump;
    dump["f"] = f;
    dump["dV"] = par->dV;
    dump["dy_volume"] = mesh->getCoordinates()->dy() * volume;
    dump["dl"] = mesh->getCoordinates()->dy() * sqrt(mesh->getCoordinates()->g_22());
    dump["div_bndry"] = par->Div_par_K_Grad_par(K, f, true);
    dump["div_nobndry"] = par->Div_par_K_Grad_par(K, f, false);
    dump["laplace_par"] = par->Div_par_Grad_par(f);
    dump["div_bndry_label"] = par->Div_par_K_Grad_par(K, f_label, true);
    dump["div_bndry_expr"] = par->Div_par_K_Grad_par(K, f_expr, true);

    // Variable coefficient, Neumann boundaries: with bndry_flux = false the
    // physical volume integral of the divergence vanishes
    Field3D Kv = FieldFactory::get()->create3D("K_variable", Options::getRoot(), mesh);
    Field3D g = FieldFactory::get()->create3D("g", Options::getRoot(), mesh);
    mesh->communicate(Kv, g);
    Kv.applyParallelBoundary("parallel_neumann_o1");
    g.applyParallelBoundary("parallel_neumann_o1");
    dump["div_variable"] = par->Div_par_K_Grad_par(Kv, g, false);

    // K2 divergence (matrices from the grid file, if present)
    dump["has_k2"] = par->hasK2() ? 1 : 0;
    if (par->hasK2()) {
      Field3D flow;
      dump["k2_div_bndry"] = par->Div_par_K_Grad_par_k2(K, f, true);
      dump["k2_div_nobndry"] = par->Div_par_K_Grad_par_k2(K, f, false);
      dump["k2_laplace_par"] = par->Div_par_Grad_par_k2(f);
      dump["k2_div_mod"] = Div_par_K_Grad_par_mod(
          K, f, flow, true, bout::ConductionMethod::PetscOperatorK2);
      dump["k2_div_variable"] = par->Div_par_K_Grad_par_k2(Kv, g, false);
      // Variable K with the Dirichlet values of f at the walls: boundary
      // fluxes enter through the boundary legs, which hold only their own
      // face, as in the support operator
      dump["k2_div_variable_bndry"] = par->Div_par_K_Grad_par_k2(Kv, f, true);
      dump["so_div_variable_bndry"] = par->Div_par_K_Grad_par(Kv, f, true);
    } else {
      // Without the matrices, both entry points must throw a clear error
      int errors = 0;
      try {
        par->Div_par_K_Grad_par_k2(K, f, true);
      } catch (const BoutException& e) {
        output.write("Expected error: {}\n", e.what());
        errors +=
            std::string(e.what()).find("k2_forward_rows") != std::string::npos ? 1 : 0;
      }
      try {
        Field3D flow;
        Div_par_K_Grad_par_mod(K, f, flow, true, bout::ConductionMethod::PetscOperatorK2);
      } catch (const BoutException& e) {
        errors +=
            std::string(e.what()).find("k2_forward_rows") != std::string::npos ? 1 : 0;
      }
      dump["k2_missing_errors"] = errors;
    }
    bout::writeDefaultOutputFile(dump);
  }

  BoutFinalise();
  return 0;
}
