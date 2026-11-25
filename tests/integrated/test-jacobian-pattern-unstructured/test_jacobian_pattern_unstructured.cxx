// Two fields on an unstructured mesh, coupled along the magnetic field by the
// support-operator conduction and across it by the perpendicular operator.
// See test_jacobian_pattern_unstructured.py

#include <bout/bout_types.hxx>
#include <bout/field3d.hxx>
#include <bout/fv_ops.hxx>
#include <bout/petsc_operators.hxx>
#include <bout/physicsmodel.hxx>
#include <bout/unused.hxx>

#include <memory>

class JacobianPattern : public PhysicsModel {
  Field3D n, T;
  std::shared_ptr<PetscOperators::Parallel> parallel;
  std::shared_ptr<FV::Div_a_Grad_perp_FCI> perpendicular;

protected:
  int init(bool UNUSED(restarting)) override {
    solver->add(n, "n");
    solver->add(T, "T");
    parallel = PetscOperators::Cache::getParallel(mesh);
    perpendicular = FV::Div_a_Grad_perp_FCI::create(mesh, 1.0);
    return 0;
  }

  int rhs(BoutReal UNUSED(time)) override {
    mesh->communicate(n, T);
    ddt(n) = parallel->Div_par_K_Grad_par(T, n) + (*perpendicular)(T, n, false);
    ddt(T) = parallel->Div_par_K_Grad_par(n, T) - (n * T);
    return 0;
  }
};

BOUTMAIN(JacobianPattern);
