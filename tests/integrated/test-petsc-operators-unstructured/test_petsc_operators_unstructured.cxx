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

    // f linear along the (straight) field lines. The boundary values (at the
    // walls, half a cell beyond the last cells) are passed as a field fb whose
    // value in each boundary cell is f at its wall. (The expression form of
    // the boundary condition evaluates expressions at an integer y index.)
    Field3D f = FieldFactory::get()->create3D("f", Options::getRoot(), mesh);
    Field3D fb = FieldFactory::get()->create3D("f_wall", Options::getRoot(), mesh);
    mesh->communicate(f);
    // (Field3D::applyParallelBoundary(region, condition, field) selects regions
    // by label, which is not unique here)
    auto* factory = BoundaryFactory::getInstance();
    for (const auto& region : mesh->getBoundariesPar()) {
      const std::unique_ptr<BoundaryOpPar> tmp{
          dynamic_cast<BoundaryOpPar*>(factory->create("parallel_dirichlet_o2", region))};
      const std::unique_ptr<BoundaryOpPar> op{tmp->clone(region.get(), &fb)};
      op->apply(f);
    }

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
    bout::writeDefaultOutputFile(dump);
  }

  BoutFinalise();
  return 0;
}
