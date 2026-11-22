#include "bout/parallel_boundary_op.hxx"
#include "bout/boundary_region_iter.hxx"
#include "bout/bout_types.hxx"
#include "bout/boutexception.hxx"
#include "bout/constants.hxx"
#include "bout/field_factory.hxx"
#include "bout/globals.hxx"
#include "bout/mesh.hxx"
#include "bout/output.hxx"

#include <cmath>

BoutReal BoundaryOpPar::getValue(const bout::boundary::BoundaryRegionFCI::Point& bndry,
                                 BoutReal t) {
  switch (value_type) {
  case ValueType::GEN: {
    Mesh* mesh = bndry.localmesh();
    bout::generator::Context context(bndry.s_x(), bndry.s_y(), bndry.s_z(), CELL_CENTRE,
                                     mesh, t);
    if (!unstructured.has_value()) {
      unstructured = mesh->isUnstructured();
    }
    if (*unstructured) {
      // The boundary point is at a real y index, half a cell beyond the last
      // cell; the integer Context truncates it (to the cell itself or to its
      // neighbour). Index the boundary cell instead, so that grid variables
      // are read in its own plane, and interpolate y to the real index.
      // (Structured meshes keep the previous behaviour.)
      // Use the boundary cell and its interior neighbour (both real cells:
      // GlobalY of a guard index can wrap around on periodic meshes)
      const BoutReal s_y = bndry.s_y();
      const int iy_cell = bndry.ind().y();
      context = bout::generator::Context(bndry.ind().x(), iy_cell, bndry.ind().z(),
                                         CELL_CENTRE, mesh, t);
      const int iy_inner = (s_y > iy_cell) ? iy_cell - 1 : iy_cell + 1;
      const BoutReal y_cell = mesh->GlobalY(iy_cell);
      const BoutReal slope = (y_cell - mesh->GlobalY(iy_inner)) / (iy_cell - iy_inner);
      context.set("y", TWOPI * (y_cell + slope * (s_y - iy_cell)));
    }
    return gen_values->generate(context);
  }
  case ValueType::FIELD:
    // FIXME: Interpolate to s_x, s_y, s_z...
    return (*field_values)[bndry.ind()];
  case ValueType::REAL:
    return real_value;
  default:
    throw BoutException("Invalid value_type encountered in BoundaryOpPar::getValue");
  }
}

BoutReal BoundaryOpPar::getValue(const bout::boundary::BoundaryRegionX::Iterator& bndry,
                                 [[maybe_unused]] BoutReal t) {
  switch (value_type) {
  case ValueType::FIELD:
    // FIXME: Interpolate to s_x, s_y, s_z...
    return (*field_values)[bndry.ind()];
  case ValueType::REAL:
    return real_value;
  default:
    throw BoutException("Invalid value_type encountered in BoundaryOpPar::getValue");
  }
}

BoutReal BoundaryOpPar::getValue(const bout::boundary::BoundaryRegionY::Iterator& bndry,
                                 [[maybe_unused]] BoutReal t) {
  switch (value_type) {
  case ValueType::FIELD:
    // FIXME: Interpolate to s_x, s_y, s_z...
    return (*field_values)[bndry.ind()];
  case ValueType::REAL:
    return real_value;
  default:
    throw BoutException("Invalid value_type encountered in BoundaryOpPar::getValue");
  }
}
