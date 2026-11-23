/*!*************************************************************************
 * \file field2d.cxx
 *
 * Class for 2D X-Y profiles
 *
 **************************************************************************
 * Copyright 2010 - 2026 BOUT++ developers
 *
 * Contact: Ben Dudson, dudson2@llnl.gov
 *
 * This file is part of BOUT++.
 *
 * BOUT++ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BOUT++ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with BOUT++.  If not, see <http://www.gnu.org/licenses/>.
 *
 **************************************************************************/

#include "bout/bout_types.hxx"
#include "bout/build_defines.hxx"

#include "bout/unused.hxx"
#include <bout/array.hxx>
#include <bout/assert.hxx>
#include <bout/boundary_factory.hxx>
#include <bout/boundary_op.hxx>
#include <bout/boutcomm.hxx>
#include <bout/boutexception.hxx>
#include <bout/field2d.hxx>
#include <bout/field3d.hxx>
#include <bout/globals.hxx> // for mesh
#include <bout/mesh.hxx>
#include <bout/output.hxx>

#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

Field2D::Field2D(Mesh* localmesh, CELL_LOC location_in, DirectionTypes directions_in,
                 [[maybe_unused]] std::optional<size_t> regionID)
    : Field(localmesh, location_in, directions_in) {
  if (fieldmesh != nullptr) {
    // Note: Even if fieldmesh is not null, LocalNx and LocalNy may
    // not be initialised.
    nx = fieldmesh->LocalNx;
    ny = fieldmesh->LocalNy;
  }
#if BOUT_USE_TRACK
  name = "<F2D>";
#endif
}

Field2D::Field2D(const Field2D& f) : Field(f), data(f.data) {
  if (fieldmesh != nullptr) {
    nx = fieldmesh->LocalNx;
    ny = fieldmesh->LocalNy;
  }
#if BOUT_USE_TRACK
  name = f.name;
#endif
}

Field2D::Field2D(BoutReal val, Mesh* localmesh) : Field2D(localmesh) { *this = val; }

Field2D::Field2D(Array<BoutReal> data_in, Mesh* localmesh, CELL_LOC datalocation,
                 DirectionTypes directions_in)
    : Field(localmesh, datalocation, directions_in), data(std::move(data_in)) {

  ASSERT1(fieldmesh != nullptr);

  nx = fieldmesh->LocalNx;
  ny = fieldmesh->LocalNy;

  ASSERT1(data.size() == nx * ny);

  setLocation(datalocation);
}

Field2D::~Field2D() { delete deriv; }

Field2D& Field2D::allocate() {
  if (data.empty()) {
    if (fieldmesh == nullptr) {
      // fieldmesh was not initialized when this field was initialized, so use
      // the global mesh
      fieldmesh = bout::globals::mesh;
    }
    // Get size from the mesh.
    nx = fieldmesh->LocalNx;
    ny = fieldmesh->LocalNy;
    ASSERT1(nx > 0);
    ASSERT1(ny > 0);
    data.reallocate(nx * ny);
#if CHECK > 2
    invalidateGuards(*this);
#endif
  } else {
    data.ensureUnique();
  }

  return *this;
}

Field2D* Field2D::timeDeriv() {
  if (deriv == nullptr) {
    deriv = new Field2D{emptyFrom(*this)};
  }
  return deriv;
}

////////////// Indexing ///////////////////

const Region<Ind2D>& Field2D::getRegion(REGION region) const {
  return fieldmesh->getRegion2D(toString(region));
}
const Region<Ind2D>& Field2D::getRegion(const std::string& region_name) const {
  return fieldmesh->getRegion2D(region_name);
}

// Not in header because we need to access fieldmesh
BoutReal& Field2D::operator[](const Ind3D& d) {
  return operator[](fieldmesh->map3Dto2D(d));
}

const BoutReal& Field2D::operator[](const Ind3D& d) const {
  return operator[](fieldmesh->map3Dto2D(d));
}

///////////// OPERATORS ////////////////

Field2D& Field2D::operator=(const Field2D& rhs) {
  // Check for self-assignment
  if (this == &rhs) {
    return (*this); // skip this assignment
  }

  Field::operator=(rhs);

  // Copy the data and data sizes
  nx = rhs.nx;
  ny = rhs.ny;

  // Copy reference to data
  data = rhs.data;

  return *this;
}

Field2D& Field2D::operator=(Field2D&& rhs) noexcept {
  // Check for self-assignment
  if (this == &rhs) {
    return (*this); // skip this assignment
  }

  // Move the data and data sizes
  nx = rhs.nx;
  ny = rhs.ny;

  // Move reference to data
  data = std::move(rhs.data);

  // Move base slice last
  Field::operator=(std::move(rhs));

  return *this;
}

Field2D& Field2D::operator=(const BoutReal rhs) {
#if BOUT_USE_TRACK
  name = "<r2D>";
#endif

  allocate();

  BOUT_FOR(i, getRegion("RGN_ALL")) { (*this)[i] = rhs; }

  return *this;
}

///////////////////// BOUNDARY CONDITIONS //////////////////

namespace {
/// Dirichlet wall nodes of an unstructured grid (see field3d.cxx)
const Region<Ind2D>& wallNodes(Mesh* mesh) {
  static const Region<Ind2D> none{};
  const std::string name = "RGN_WALL_NODES";
  if (mesh->hasRegion2D(name)) {
    return mesh->getRegion2D(name);
  }
  if (!mesh->isUnstructured()) {
    return none;
  }
  Field3D cell_number{mesh};
  cell_number = -1.0;
  if (mesh->sourceHasVar("cell_number")) {
    mesh->get(cell_number, "cell_number", -1.0, false);
  }
  Region<Ind2D>::RegionIndices indices;
  for (const auto* side : {"RGN_INNER_X", "RGN_OUTER_X"}) {
    BOUT_FOR_SERIAL(i, mesh->getRegion2D(side)) {
      if (cell_number(i.x(), i.y(), 0) >= 0) {
        indices.push_back(i);
      }
    }
  }
  mesh->addRegion2D(name, Region<Ind2D>(indices));
  return mesh->getRegion2D(name);
}

/// Keep the wall-node values during boundary conditions applied by name,
/// except dirichlet_node (see field3d.cxx)
class KeepWallNodes {
public:
  KeepWallNodes(Field2D& field, const std::string& condition) : field(field) {
    if (condition.rfind("dirichlet_node", 0) == 0) {
      return;
    }
    const auto& region = wallNodes(field.getMesh());
    indices = region.getIndices();
    values.reserve(indices.size());
    for (const auto& i : indices) {
      values.push_back(field[i]);
    }
  }
  KeepWallNodes(const KeepWallNodes&) = delete;
  KeepWallNodes& operator=(const KeepWallNodes&) = delete;
  ~KeepWallNodes() {
    for (std::size_t k = 0; k < indices.size(); ++k) {
      field[indices[k]] = values[k];
    }
  }

private:
  Field2D& field;
  Region<Ind2D>::RegionIndices indices;
  std::vector<BoutReal> values;
};
} // namespace

void Field2D::applyBoundary(bool init) {

#if CHECK > 0
  if (init) {
    if (not isBoundarySet()) {
      output_warn << "WARNING: Call to Field2D::applyBoundary(), but no boundary set\n";
    }
  }
#endif

  checkData(*this);

  for (const auto& bndry : getBoundaryOps()) {
    // Always apply to the values when initialising
    // fields, otherwise apply only if wanted
    if (!bndry->apply_to_ddt || init) {
      bndry->apply(*this);
    }
  }
}

void Field2D::applyBoundary(BoutReal time) {

#if CHECK > 0
  if (not isBoundarySet()) {
    output_warn << "WARNING: Call to Field2D::applyBoundary(time), but no boundary set\n";
  }
#endif

  checkData(*this);

  for (const auto& bndry : getBoundaryOps()) {
    bndry->apply(*this, time);
  }
}

void Field2D::applyBoundary(const std::string& condition) {

  checkData(*this);
  const KeepWallNodes keep_wall_nodes(*this, condition);

  /// Get the boundary factory (singleton)
  BoundaryFactory* bfact = BoundaryFactory::getInstance();

  /// Loop over the mesh boundary regions
  for (const auto& reg : fieldmesh->getBoundaries()) {
    auto op = std::unique_ptr<BoundaryOp>{
        dynamic_cast<BoundaryOp*>(bfact->create(condition, reg))};
    op->apply(*this);
  }

  // Set the corners to zero
  for (int jx = 0; jx < fieldmesh->xstart; jx++) {
    for (int jy = 0; jy < fieldmesh->ystart; jy++) {
      operator()(jx, jy) = 0.;
    }
    for (int jy = fieldmesh->yend + 1; jy < fieldmesh->LocalNy; jy++) {
      operator()(jx, jy) = 0.;
    }
  }
  for (int jx = fieldmesh->xend + 1; jx < fieldmesh->LocalNx; jx++) {
    for (int jy = 0; jy < fieldmesh->ystart; jy++) {
      operator()(jx, jy) = 0.;
    }
    for (int jy = fieldmesh->yend + 1; jy < fieldmesh->LocalNy; jy++) {
      operator()(jx, jy) = 0.;
    }
  }
}

void Field2D::applyBoundary(const std::string& region, const std::string& condition) {

  checkData(*this);
  const KeepWallNodes keep_wall_nodes(*this, condition);

  /// Get the boundary factory (singleton)
  BoundaryFactory* bfact = BoundaryFactory::getInstance();

  bool region_found = false;
  /// Loop over the mesh boundary regions
  for (const auto& reg : fieldmesh->getBoundaries()) {
    if (reg->label == region) {
      region_found = true;
      auto op = std::unique_ptr<BoundaryOp>{
          dynamic_cast<BoundaryOp*>(bfact->create(condition, reg))};
      op->apply(*this);
      break;
    }
  }

  if (!region_found) {
    throw BoutException("Region '{:s}' not found", region);
  }

  // Set the corners to zero
  for (int jx = 0; jx < fieldmesh->xstart; jx++) {
    for (int jy = 0; jy < fieldmesh->ystart; jy++) {
      operator()(jx, jy) = 0.;
    }
    for (int jy = fieldmesh->yend + 1; jy < fieldmesh->LocalNy; jy++) {
      operator()(jx, jy) = 0.;
    }
  }
  for (int jx = fieldmesh->xend + 1; jx < fieldmesh->LocalNx; jx++) {
    for (int jy = 0; jy < fieldmesh->ystart; jy++) {
      operator()(jx, jy) = 0.;
    }
    for (int jy = fieldmesh->yend + 1; jy < fieldmesh->LocalNy; jy++) {
      operator()(jx, jy) = 0.;
    }
  }
}

void Field2D::applyTDerivBoundary() {

  checkData(*this);
  ASSERT1(deriv != nullptr);
  checkData(*deriv);

  for (const auto& bndry : getBoundaryOps()) {
    bndry->apply_ddt(*this);
  }
}

void Field2D::setBoundaryTo(const Field2D& f2d) {

  checkData(f2d);

  allocate(); // Make sure data allocated

  /// Loop over boundary regions
  for (const auto& regnew : fieldmesh->getBoundaries()) {
    /// Loop within each region
    auto* reg = regnew->getLegacyPointer();
    for (reg->first(); !reg->isDone(); reg->next()) {
      // Get value half-way between cells
      BoutReal val =
          0.5 * (f2d(reg->x, reg->y) + f2d(reg->x - reg->bx, reg->y - reg->by));
      // Set to this value
      (*this)(reg->x, reg->y) = 2. * val - (*this)(reg->x - reg->bx, reg->y - reg->by);
    }
  }
}

void Field2D::swapData(Field2D& other) { std::swap(data, other.data); }

////////////// NON-MEMBER OVERLOADED OPERATORS //////////////

//////////////// NON-MEMBER FUNCTIONS //////////////////

namespace {
// Internal routine to avoid ugliness with interactions between CHECK
// levels and UNUSED parameters
#if CHECK > 2
void checkDataIsFiniteOnRegion(const Field2D& f, const std::string& region) {
  // Do full checks
  BOUT_FOR_SERIAL(i, f.getRegion(region)) {
    if (!std::isfinite(f[i])) {
      throw BoutException("Field2D: Operation on non-finite data at [{:d}][{:d}]\n",
                          i.x(), i.y());
    }
  }
}
#elif CHECK > 0
// No-op for no checking
void checkDataIsFiniteOnRegion(const Field2D& UNUSED(f),
                               const std::string& UNUSED(region)) {}
#endif
} // namespace

#if CHECK > 0
/// Check if the data is valid
void checkData(const Field2D& f, const std::string& region) {
  if (!f.isAllocated()) {
    throw BoutException("Field2D: Operation on empty data\n");
  }

  checkDataIsFiniteOnRegion(f, region);
}
#endif

#if CHECK > 2
void invalidateGuards(Field2D& var) {
  BOUT_FOR(i, var.getRegion("RGN_GUARDS")) { var[i] = BoutNaN; }
}
#endif

bool operator==(const Field2D& a, const Field2D& b) {
  if (!a.isAllocated() || !b.isAllocated()) {
    return false;
  }
  return min(abs(a - b)) < 1e-10;
}

std::ostream& operator<<(std::ostream& out, const Field2D& value) {
  out << toString(value);
  return out;
}

void swap(Field2D& first, Field2D& second) noexcept {
  using std::swap;

  // Swap base class members
  swap(static_cast<Field&>(first), static_cast<Field&>(second));

  swap(first.data, second.data);
  swap(first.nx, second.nx);
  swap(first.ny, second.ny);
  swap(first.deriv, second.deriv);
}
