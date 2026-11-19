// Support operators on an unstructured grid with parallel (y) walls:
// cell volume and boundary fluxes. See runtest.

#include "bout/boundary_factory.hxx"
#include "bout/boundary_region_iter.hxx"
#include "bout/bout.hxx"
#include "bout/difops.hxx"
#include "bout/field3d.hxx"
#include "bout/field_factory.hxx"
#include "bout/globals.hxx"
#include "bout/options.hxx"
#include "bout/options_io.hxx"
#include "bout/parallel_boundary_op.hxx"
#include "bout/petsc_operators.hxx"

#include <memory>

int main(int argc, char** argv) {
  BoutInitialise(argc, argv);

  {
    using bout::globals::mesh;
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
    bout::writeDefaultOutputFile(dump);
  }

  BoutFinalise();
  return 0;
}
