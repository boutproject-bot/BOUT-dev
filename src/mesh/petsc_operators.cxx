#include "bout/build_defines.hxx"

#if BOUT_HAS_PETSC

#include "bout/assert.hxx"
#include "bout/bout_types.hxx"
#include "bout/boutexception.hxx"
#include "bout/coordinates.hxx"
#include "bout/field2d.hxx"
#include "bout/field3d.hxx"
#include "bout/output.hxx"
#include "bout/output_bout_types.hxx"
#include "bout/petsc_interface.hxx"
#include "bout/petsc_operators.hxx"
#include "bout/petsclib.hxx"
#include "bout/region.hxx"
#include "bout/utils.hxx"

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

PetscIndexMapping::~PetscIndexMapping() {
  if (mat_stored_to_petsc != nullptr) {
    BOUT_DO_PETSC(MatDestroy(&mat_stored_to_petsc));
  }
  if (mat_petsc_to_stored != nullptr) {
    BOUT_DO_PETSC(MatDestroy(&mat_petsc_to_stored));
  }
}

PetscInt PetscIndexMapping::storedToPetsc(int stored_index) const {
  auto it = local_stored_to_petsc.find(stored_index);
  if (it == local_stored_to_petsc.end()) {
    throw BoutException("Stored index {} is not owned locally", stored_index);
  }
  return it->second;
}

void PetscIndexMapping::buildPermutation(PetscInt nlocal, PetscInt nglobal,
                                         const std::vector<int>& stored_indices) {
  global_size = nglobal;
  local_stored_indices = stored_indices;

  // Renumbering matrix.
  // Maps PETSc row (or column) indices to the global indices stored in
  // the mesh.  This is needed because the PETSc indices depend on the
  // number of processors.
  BOUT_DO_PETSC(MatCreate(BoutComm::get(), &mat_stored_to_petsc));
  BOUT_DO_PETSC(MatSetSizes(mat_stored_to_petsc, nlocal, nlocal, nglobal, nglobal));
  BOUT_DO_PETSC(MatSetType(mat_stored_to_petsc, MATMPIAIJ));

  // Each row will have one non-zero entry, which could be in
  // either the "diagonal" or "off-diagonal" block.
  BOUT_DO_PETSC(MatMPIAIJSetPreallocation(mat_stored_to_petsc, 1, nullptr, 1, nullptr));

  // Get the range of global rows owned by this processor
  BOUT_DO_PETSC(MatGetOwnershipRange(mat_stored_to_petsc, &row_start, &row_end));
  ASSERT1(row_end - row_start == nlocal);

  const PetscScalar one = 1.0;
  for (PetscInt i = 0; i < nlocal; ++i) {
    const PetscInt petsc_row = row_start + i;
    const PetscInt stored_col = stored_indices[i];
    ASSERT1(stored_col >= 0);
    local_stored_to_petsc[static_cast<int>(stored_col)] = petsc_row;
    BOUT_DO_PETSC(MatSetValues(mat_stored_to_petsc, 1, &petsc_row, 1, &stored_col, &one,
                               INSERT_VALUES));
  }
  BOUT_DO_PETSC(MatAssemblyBegin(mat_stored_to_petsc, MAT_FINAL_ASSEMBLY));
  BOUT_DO_PETSC(MatAssemblyEnd(mat_stored_to_petsc, MAT_FINAL_ASSEMBLY));
  // Transpose to map petsc indices to stored mesh indices
  BOUT_DO_PETSC(
      MatTranspose(mat_stored_to_petsc, MAT_INITIAL_MATRIX, &mat_petsc_to_stored));
}

Region<Ind3D> PetscCellMapping::create_region(const Field3D& cell_number) {
  Region<Ind3D>::RegionIndices indices;
  BOUT_FOR_SERIAL(i, cell_number.getRegion("RGN_NOBNDRY")) {
    if (cell_number[i] >= 0) {
      indices.push_back(i);
    }
  }
  return Region<Ind3D>(indices);
}

Region<Ind3D> PetscCellMapping::create_region_xin(const Field3D& cell_number) {
  Region<Ind3D>::RegionIndices indices;
  const auto& region = cell_number.getRegion("RGN_INNER_X");
  BOUT_FOR_SERIAL(i, region) {
    if (cell_number[i] >= 0) {
      indices.push_back(i);
    }
  }
  return Region<Ind3D>(indices);
}

Region<Ind3D> PetscCellMapping::create_region_xout(const Field3D& cell_number) {
  Region<Ind3D>::RegionIndices indices;
  const auto& region = cell_number.getRegion("RGN_OUTER_X");
  BOUT_FOR_SERIAL(i, region) {
    if (cell_number[i] >= 0) {
      indices.push_back(i);
    }
  }
  return Region<Ind3D>(indices);
}

PetscCellMapping::PetscCellMapping(const Field3D& cell_number,
                                   const Field3D& forward_cell_number,
                                   const Field3D& backward_cell_number, int total_cells)
    : cell_number(cell_number), forward_cell_number(forward_cell_number),
      backward_cell_number(backward_cell_number),
      evolving_region(create_region(cell_number)),
      xin_region(create_region_xin(cell_number)),
      xout_region(create_region_xout(cell_number)),
      yup_region(create_region(forward_cell_number)),
      ydown_region(create_region(backward_cell_number)) {
  std::vector<int> local_indices;
  const auto push_region = [&](const Region<Ind3D>& region, const Field3D& values) {
    BOUT_FOR_SERIAL(i, region) { local_indices.push_back(ROUND(values[i])); }
  };
  push_region(evolving_region, this->cell_number);
  push_region(xin_region, this->cell_number);
  push_region(xout_region, this->cell_number);
  push_region(yup_region, this->forward_cell_number);
  push_region(ydown_region, this->backward_cell_number);
  buildPermutation(static_cast<PetscInt>(local_indices.size()), total_cells,
                   local_indices);
}

IS PetscCellMapping::makeEvolvingIS() const {
  // Collect global PETSc indices in mapOwnedInteriorCells order.
  // Reserve the known count up front to avoid reallocation.
  std::vector<PetscInt> indices;
  indices.reserve(static_cast<std::size_t>(evolving_region.size()));

  mapOwnedInteriorCells(
      [&](PetscInt row, const Ind3D& /*i*/, int /*stored*/) { indices.push_back(row); });

  IS is = nullptr;
  BOUT_DO_PETSC(ISCreateGeneral(BoutComm::get(), static_cast<PetscInt>(indices.size()),
                                indices.data(), PETSC_COPY_VALUES, &is));
  return is;
}

Mat PetscCellMapping::extractEvolvingSubmatrix(
    const PetscOperator<CellSpaceTag, CellSpaceTag>& op) const {
  IS is = makeEvolvingIS();

  Mat sub = nullptr;
  BOUT_DO_PETSC(MatCreateSubMatrix(op.raw(), is, is, MAT_INITIAL_MATRIX, &sub));

  BOUT_DO_PETSC(ISDestroy(&is));

  return sub;
}

PetscLegMapping::PetscLegMapping(int total_legs, std::vector<int> local_leg_indices) {
  std::ranges::sort(local_leg_indices);
  local_leg_indices.erase(std::unique(local_leg_indices.begin(), local_leg_indices.end()),
                          local_leg_indices.end());
  buildPermutation(static_cast<PetscInt>(local_leg_indices.size()), total_legs,
                   local_leg_indices);
}

PetscCellVector makePetscCellVector(const PetscCellMappingPtr& mapping,
                                    const Field3D& field) {
  ASSERT1(field.hasParallelSlices());
  ASSERT1(field.yup().isAllocated());
  ASSERT1(field.ydown().isAllocated());
  PetscCellVector out(mapping);
  PetscScalar* x{nullptr};
  BOUT_DO_PETSC(VecGetArray(out.raw(), &x));
  mapping->mapLocalField([&](PetscInt row, const Ind3D& i) { x[row] = field[i]; });
  const auto& yup = field.yup();
  mapping->mapLocalYup([&](PetscInt row, const Ind3D& i) { x[row] = yup[i]; });
  const auto& ydown = field.ydown();
  mapping->mapLocalYdown([&](PetscInt row, const Ind3D& i) { x[row] = ydown[i]; });
  BOUT_DO_PETSC(VecRestoreArray(out.raw(), &x));
  return out;
}

Field3D toField3D(const PetscCellVector& vec, const Field3D& prototype) {
  auto mapping = std::static_pointer_cast<const PetscCellMapping>(vec.getMapping());
  Field3D result{emptyFrom(prototype)};
  result.splitParallelSlices();
  result.yup().allocate();
  result.ydown().allocate();

  const PetscScalar* x{nullptr};
  BOUT_DO_PETSC(VecGetArrayRead(vec.raw(), &x));
  mapping->mapLocalField([&](PetscInt row, const Ind3D& i) { result[i] = x[row]; });
  mapping->mapLocalYup([&](PetscInt row, const Ind3D& i) { result.yup()[i] = x[row]; });
  mapping->mapLocalYdown(
      [&](PetscInt row, const Ind3D& i) { result.ydown()[i] = x[row]; });
  BOUT_DO_PETSC(VecRestoreArrayRead(vec.raw(), &x));

  return result;
}

Field3D PetscOperators::meshGetField3D(Mesh* mesh, const std::string& name) {
  Field3D result;
  if (mesh->get(result, name, 0.0, false) != 0) {
    throw BoutException("PetscOperators requires field '{}'", name);
  }
  return result;
}

int PetscOperators::meshGetInt(const std::string& name) const {
  int result{0};
  if (mesh->get(result, name) != 0) {
    throw BoutException("PetscOperators requires int '{}'", name);
  }
  return result;
}

PetscOperators::PetscOperators(Mesh* mesh)
    : mesh(mesh),
      cell_mapping(std::make_shared<PetscCellMapping>(
          meshGetField3D(mesh, "cell_number"),
          meshGetField3D(mesh, "forward_cell_number"),
          meshGetField3D(mesh, "backward_cell_number"), meshGetInt("total_cells"))),
      forward_leg_interior_number(meshGetField3D(mesh, "forward_leg_interior_number")),
      forward_leg_boundary_number(meshGetField3D(mesh, "forward_leg_boundary_number")),
      backward_leg_interior_number(meshGetField3D(mesh, "backward_leg_interior_number")),
      backward_leg_boundary_number(meshGetField3D(mesh, "backward_leg_boundary_number")) {
  std::vector<int> local_forward_legs;
  std::vector<int> local_backward_legs;
  cell_mapping->mapOwnedInteriorCells([&](PetscInt, const Ind3D& i, int) {
    const int f_int = ROUND(forward_leg_interior_number[i]);
    const int f_bnd = ROUND(forward_leg_boundary_number[i]);
    const int b_int = ROUND(backward_leg_interior_number[i]);
    const int b_bnd = ROUND(backward_leg_boundary_number[i]);
    if (f_int >= 0) {
      local_forward_legs.push_back(f_int);
    }
    if (f_bnd >= 0) {
      local_forward_legs.push_back(f_bnd);
    }
    if (b_int >= 0) {
      local_backward_legs.push_back(b_int);
    }
    if (b_bnd >= 0) {
      local_backward_legs.push_back(b_bnd);
    }
  });

  forward_leg_mapping = std::make_shared<PetscLegMapping>(meshGetInt("n_forward_legs"),
                                                          std::move(local_forward_legs));
  backward_leg_mapping = std::make_shared<PetscLegMapping>(
      meshGetInt("n_backward_legs"), std::move(local_backward_legs));
}

PetscForwardOperator PetscOperators::forward() const {
  return PetscForwardOperator(
      forward_leg_mapping, cell_mapping, meshGetArray<int>("forward_rows"),
      meshGetArray<int>("forward_columns"), meshGetArray<BoutReal>("forward_weights"));
}

PetscBackwardOperator PetscOperators::backward() const {
  return PetscBackwardOperator(
      backward_leg_mapping, cell_mapping, meshGetArray<int>("backward_rows"),
      meshGetArray<int>("backward_columns"), meshGetArray<BoutReal>("backward_weights"));
}

template <typename LegTag>
PetscOperator<LegTag, CellSpaceTag>
PetscOperators::buildInjection(const Field3D& interior_leg_number,
                               const Field3D& boundary_leg_number,
                               std::shared_ptr<const PetscLegMapping> leg_mapping) const {
  using Op = PetscOperator<LegTag, CellSpaceTag>;
  bout::petsc::UniqueMat mat{new Mat{nullptr}};
  BOUT_DO_PETSC(MatCreate(BoutComm::get(), mat.get()));
  BOUT_DO_PETSC(MatSetType(*mat, MATMPIAIJ));
  BOUT_DO_PETSC(MatSetSizes(*mat, leg_mapping->localSize(), cell_mapping->localSize(),
                            leg_mapping->globalSize(), cell_mapping->globalSize()));
  BOUT_DO_PETSC(MatMPIAIJSetPreallocation(*mat, 1, nullptr, 1, nullptr));
  const PetscScalar one = 1.0;
  cell_mapping->mapOwnedInteriorCells([&](PetscInt cell_row, const Ind3D& i, int) {
    const int interior = ROUND(interior_leg_number[i]);
    const int boundary = ROUND(boundary_leg_number[i]);
    if (interior >= 0) {
      const PetscInt leg_row = leg_mapping->storedToPetsc(interior);
      BOUT_DO_PETSC(MatSetValues(*mat, 1, &leg_row, 1, &cell_row, &one, INSERT_VALUES));
    }
    if (boundary >= 0) {
      const PetscInt leg_row = leg_mapping->storedToPetsc(boundary);
      BOUT_DO_PETSC(MatSetValues(*mat, 1, &leg_row, 1, &cell_row, &one, INSERT_VALUES));
    }
  });
  BOUT_DO_PETSC(MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY));
  BOUT_DO_PETSC(MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY));
  return Op(leg_mapping, cell_mapping, std::move(mat));
}

template PetscForwardOperator PetscOperators::buildInjection<ForwardLegSpaceTag>(
    const Field3D&, const Field3D&, std::shared_ptr<const PetscLegMapping>) const;
template PetscBackwardOperator PetscOperators::buildInjection<BackwardLegSpaceTag>(
    const Field3D&, const Field3D&, std::shared_ptr<const PetscLegMapping>) const;

PetscForwardOperator PetscOperators::injectForward() const {
  return buildInjection<ForwardLegSpaceTag>(
      forward_leg_interior_number, forward_leg_boundary_number, forward_leg_mapping);
}

PetscBackwardOperator PetscOperators::injectBackward() const {
  return buildInjection<BackwardLegSpaceTag>(
      backward_leg_interior_number, backward_leg_boundary_number, backward_leg_mapping);
}

PetscForwardLegVector PetscOperators::forwardLegWeights() const {
  return loadLegWeights<PetscForwardLegVector>("forward_leg_weights",
                                               forward_leg_mapping);
}

PetscBackwardLegVector PetscOperators::backwardLegWeights() const {
  return loadLegWeights<PetscBackwardLegVector>("backward_leg_weights",
                                                backward_leg_mapping);
}

PetscCellOperator PetscOperators::diagonal(const Field3D& f) const {
  return PetscCellOperator::diagonal(cell_mapping, makePetscCellVector(cell_mapping, f));
}

PetscForwardLegOperator PetscOperators::diagonal(const PetscForwardLegVector& v) const {
  return PetscForwardLegOperator::diagonal(forward_leg_mapping, v);
}

PetscBackwardLegOperator PetscOperators::diagonal(const PetscBackwardLegVector& v) const {
  return PetscBackwardLegOperator::diagonal(backward_leg_mapping, v);
}

PetscOperators::Parallel PetscOperators::getParallel() const {
  // Set boundaries
  const PetscCellOperator& Neumann_fwd =
      makeNeumannOperator(cell_mapping, BoundaryDirection::Forward); // C <- C
  const PetscCellOperator& Neumann_bwd =
      makeNeumannOperator(cell_mapping, BoundaryDirection::Backward); // C <- C

  // Primitive operators from the mesh / metadata
  auto Forward = this->forward();             // L+ <- C
  auto Backward = this->backward();           // L- <- C
  auto Inject_plus = this->injectForward();   // L+ <- C
  auto Inject_minus = this->injectBackward(); // L- <- C

  // Leg weights
  const auto w_plus_vec = forwardLegWeights();
  const auto w_minus_vec = backwardLegWeights();
  const auto W_plus = diagonal(w_plus_vec);   // L+ <- L+
  const auto W_minus = diagonal(w_minus_vec); // L- <- L-

  // Weighted adjoints: C <- L+ and C <- L-
  // The Neumann_f/bwd operators set the boundary cells to which the legs are connected
  auto Restrict_minus = Neumann_fwd * Inject_plus.transpose() * W_plus;
  auto Restrict_plus = Neumann_bwd * Inject_minus.transpose() * W_minus;

  auto* coords = mesh->getCoordinates();

  // Parallel spacing in cell space
  Field3D dl = Coordinates::FieldMetric{coords->dy() * sqrt(coords->g_22())};
  dl.splitParallelSlices();
  dl.yup() = 0.0;
  dl.ydown() = 0.0;
  dl.applyParallelBoundary("parallel_neumann_o1");

  // Cell volume
  Field3D dV =
      Coordinates::FieldMetric{coords->J() * coords->dx() * coords->dy() * coords->dz()};
  if (mesh->isUnstructured() and mesh->sourceHasVar("dagp_us_volume")) {
    // Unstructured grids use dx = dz = 1 and store the poloidal volume
    // element of each node (its dual cell) separately: the area, times the
    // major radius on cylindrical grids. The physical volume is dy times
    // this, the measure that the perpendicular operator conserves. J is not
    // used: it is sqrt(g_22) on these grids, which would count R twice.
    dV = Coordinates::FieldMetric{coords->dy()} * meshGetField3D(mesh, "dagp_us_volume");
  }
  dV.splitParallelSlices();
  dV.yup() = 0.0;
  dV.ydown() = 0.0;
  dV.applyParallelBoundary("parallel_neumann_o1");

  Field3D neg_inv_dV = -1.0 / dV;
  neg_inv_dV.splitParallelSlices();
  neg_inv_dV.yup() = 0.0;
  neg_inv_dV.ydown() = 0.0;
  neg_inv_dV.applyParallelBoundary("parallel_neumann_o1");

  const auto DV = diagonal(dV);
  const auto Neg_inv_dV = diagonal(neg_inv_dV);

  // Unweighted interpolation to +1/2 and -1/2 legs
  auto Interp_plus = (Forward + Inject_plus) * 0.5;    // L+ <- C
  auto Interp_minus = (Backward + Inject_minus) * 0.5; // L- <- C

  // Leg-centered dl
  const auto dl_plus = Interp_plus(dl);
  const auto dl_minus = Interp_minus(dl);

  const auto inv_dl_plus = PetscForwardLegVector::reciprocal(dl_plus);
  const auto inv_dl_minus = PetscBackwardLegVector::reciprocal(dl_minus);

  const auto Inv_dl_plus = diagonal(inv_dl_plus);   // L+ <- L+
  const auto Inv_dl_minus = diagonal(inv_dl_minus); // L- <- L-

  // Half-step gradients
  auto Grad_plus = Inv_dl_plus * (Forward - Inject_plus);     // L+ <- C
  auto Grad_minus = Inv_dl_minus * (Inject_minus - Backward); // L- <- C

  // Cell-centered central gradient
  // Apply Neumann so that the yup/ydown cells are set. This is needed
  // so that Div_par has the correct dependence on yup/down cells.
  auto Grad_par =
      ((Restrict_minus * Grad_plus) + (Restrict_plus * Grad_minus)) * 0.5; // C <- C

  // Leg-centered volumes from unweighted interpolation
  const auto dV_plus = Interp_plus(dV);   // L+
  const auto dV_minus = Interp_minus(dV); // L-

  // Physical leg volumes include the leg weights
  const auto dV_leg_plus = PetscForwardLegVector::pointwiseMultiply(dV_plus, w_plus_vec);
  const auto dV_leg_minus =
      PetscBackwardLegVector::pointwiseMultiply(dV_minus, w_minus_vec);

  const auto DV_leg_plus = diagonal(dV_leg_plus);   // L+ <- L+
  const auto DV_leg_minus = diagonal(dV_leg_minus); // L- <- L-

  // Support-operator divergence from half-step gradients
  auto Div_minus = Neg_inv_dV * Grad_plus.transpose() * DV_leg_plus;  // C <- L+
  auto Div_plus = Neg_inv_dV * Grad_minus.transpose() * DV_leg_minus; // C <- L-

  // Cell-centered divergence paired with the cell-centered gradient
  auto Div_par = Neg_inv_dV * Grad_par.transpose() * DV; // C <- C

  // Boundary virtual cells own no legs, so a face at a parallel boundary
  // appears in only one of the two half-operators (as the leg from the last
  // cell into the boundary cell), while every interior face appears in
  // both. Boundary legs therefore get weight 2 in their half-operator, so
  // that the average of the two halves contains the full boundary flux.
  const auto Bndry_weight_plus = diagonal(legValues<PetscForwardLegVector>(
      forward_leg_boundary_number, forward_leg_mapping, 1.0, 2.0));
  const auto Bndry_weight_minus = diagonal(legValues<PetscBackwardLegVector>(
      backward_leg_boundary_number, backward_leg_mapping, 1.0, 2.0));

  // Diffusion operator assembled from the half-step pieces
  auto Div_par_Grad_par = ((Div_minus * Bndry_weight_plus * Grad_plus)
                           + (Div_plus * Bndry_weight_minus * Grad_minus))
                          * 0.5; // C <- C

  Parallel par{// Cell <- Cell operators
               std::move(Grad_par), std::move(Div_par), std::move(Div_par_Grad_par),
               std::move(dV),

               // Leg <-> Cell operators
               std::move(Grad_minus), std::move(Grad_plus), std::move(Div_minus),
               std::move(Div_plus),

               std::move(Backward), std::move(Forward), std::move(Inject_minus),
               std::move(Inject_plus),

               std::move(Interp_minus), std::move(Interp_plus),

               std::move(Restrict_minus), std::move(Restrict_plus),

               interiorLegMask<PetscForwardLegVector>(forward_leg_boundary_number,
                                                      forward_leg_mapping),
               interiorLegMask<PetscBackwardLegVector>(backward_leg_boundary_number,
                                                       backward_leg_mapping),

               // K2 divergence: read below if the grid has it
               std::nullopt, std::nullopt};

  if (mesh->sourceHasVar("k2_forward_weights")
      or mesh->sourceHasVar("k2_backward_weights")) {
    // The K2 weights contain 1 / dl+- computed by the grid generator from the
    // metric in the grid file (dl = dy * sqrt(g_22), SI units), whereas the
    // leg fluxes use the parallel lengths of this mesh, which may be
    // normalised (Hermes-3 divides g_22 by rho_s0^2 on FCI grids). Convert
    // the K2 columns to the lengths of this mesh: factor dl_file / dl_mesh
    // on each leg (1 if the metric is used unchanged).
    Field3D dl_file = dl;
    if (mesh->sourceHasVar("dy") and mesh->sourceHasVar("g_22")) {
      dl_file = meshGetField3D(mesh, "dy") * sqrt(meshGetField3D(mesh, "g_22"));
    } else {
      output_warn.write("PetscOperators: dy or g_22 missing in the grid file; the K2 "
                        "matrices are used with the lengths of this mesh\n");
    }
    dl_file.splitParallelSlices();
    dl_file.yup() = 0.0;
    dl_file.ydown() = 0.0;
    dl_file.applyParallelBoundary("parallel_neumann_o1");
    loadK2(
        par, PetscForwardLegVector::pointwiseMultiply(dV_leg_plus, inv_dl_plus),
        PetscBackwardLegVector::pointwiseMultiply(dV_leg_minus, inv_dl_minus),
        PetscForwardLegVector::pointwiseMultiply(par.Interp_plus(dl_file), inv_dl_plus),
        PetscBackwardLegVector::pointwiseMultiply(par.Interp_minus(dl_file),
                                                  inv_dl_minus));
  }
  return par;
}

namespace {
/// max |v| over the legs where mask is 1
template <typename LegVector>
BoutReal maskedMaxAbs(const LegVector& v, const LegVector& mask) {
  const auto masked = LegVector::pointwiseMultiply(v, mask);
  PetscReal result{0.0};
  BOUT_DO_PETSC(VecNorm(masked.raw(), NORM_INFINITY, &result));
  return result;
}

template <typename LegVector>
BoutReal maxAbs(const LegVector& v) {
  PetscReal result{0.0};
  BOUT_DO_PETSC(VecNorm(v.raw(), NORM_INFINITY, &result));
  return result;
}
} // namespace

template <typename Operator>
Operator PetscOperators::readK2Operator(const std::string& direction,
                                        const PetscLegMappingPtr& leg_mapping) const {
  const auto rows = meshGetArray<int>(fmt::format("k2_{}_rows", direction));
  const auto columns = meshGetArray<int>(fmt::format("k2_{}_columns", direction));
  const auto weights = meshGetArray<BoutReal>(fmt::format("k2_{}_weights", direction));
  const int n_rows = meshGetInt("total_cells");
  const int n_legs = leg_mapping->globalSize();
  if (static_cast<int>(rows.size()) != n_rows + 1) {
    throw BoutException("k2_{}_rows has {} entries, expected total_cells + 1 = {}",
                        direction, rows.size(), n_rows + 1);
  }
  if (columns.size() != weights.size() or rows[0] != 0
      or rows[n_rows] != static_cast<int>(columns.size())) {
    throw BoutException("k2_{}_rows/columns/weights are not a consistent CSR matrix",
                        direction);
  }
  for (const int column : columns) {
    if (column < 0 or column >= n_legs) {
      throw BoutException("k2_{}_columns: leg index {} outside [0, n_{}_legs = {})",
                          direction, column, direction, n_legs);
    }
  }
  return Operator(cell_mapping, leg_mapping, rows, columns, weights);
}

void PetscOperators::loadK2(Parallel& par, const PetscForwardLegVector& scale_plus,
                            const PetscBackwardLegVector& scale_minus,
                            const PetscForwardLegVector& length_plus,
                            const PetscBackwardLegVector& length_minus) const {
  constexpr int supported_version = 1;
  if (mesh->sourceHasVar("k2_version")) {
    int version{0};
    mesh->get(version, "k2_version", 0);
    if (version < 1 or version > supported_version) {
      throw BoutException("K2 matrices of version {} in the grid file; this BOUT++ "
                          "supports version {}",
                          version, supported_version);
    }
  }
  // Columns scaled from the lengths of the grid file to those of this mesh
  par.Div_minus_k2 =
      readK2Operator<PetscForwardToCellOperator>("forward", forward_leg_mapping)
      * diagonal(length_plus);
  par.Div_plus_k2 =
      readK2Operator<PetscBackwardToCellOperator>("backward", backward_leg_mapping)
      * diagonal(length_minus);

  // Conservation: every interior leg flux leaves one cell and enters others,
  // so the dV-weighted column sums vanish on interior legs:
  // (dV^T Div_k2)_l = 0. This fails if the generator used another cell
  // volume (dy * dagp_us_volume) than this grid provides. The scale is the
  // own-face entry of a leg, dV_leg / dl (the support-operator column).
  const auto dV = makePetscCellVector(cell_mapping, par.dV);
  PetscForwardLegVector colsum_plus(forward_leg_mapping);
  PetscBackwardLegVector colsum_minus(backward_leg_mapping);
  BOUT_DO_PETSC(MatMultTranspose(par.Div_minus_k2->raw(), dV.raw(), colsum_plus.raw()));
  BOUT_DO_PETSC(MatMultTranspose(par.Div_plus_k2->raw(), dV.raw(), colsum_minus.raw()));
  const BoutReal error_plus =
      maskedMaxAbs(colsum_plus, par.Interior_plus) / maxAbs(scale_plus);
  const BoutReal error_minus =
      maskedMaxAbs(colsum_minus, par.Interior_minus) / maxAbs(scale_minus);
  constexpr BoutReal tolerance = 1e-10;
  if (not(error_plus < tolerance and error_minus < tolerance)) {
    throw BoutException(
        "The K2 divergence matrices of the grid file are not conservative with the "
        "cell volume dy * dagp_us_volume of this grid: relative dV-weighted column "
        "sums {:e} (forward), {:e} (backward), tolerance {:e}. Were they generated "
        "for this grid?",
        error_plus, error_minus, tolerance);
  }
  PetscReal length_min{0.0};
  PetscReal length_max{0.0};
  PetscReal length_min_minus{0.0};
  PetscReal length_max_minus{0.0};
  BOUT_DO_PETSC(VecMin(length_plus.raw(), nullptr, &length_min));
  BOUT_DO_PETSC(VecMax(length_plus.raw(), nullptr, &length_max));
  BOUT_DO_PETSC(VecMin(length_minus.raw(), nullptr, &length_min_minus));
  BOUT_DO_PETSC(VecMax(length_minus.raw(), nullptr, &length_max_minus));
  length_min = std::min(length_min, length_min_minus);
  length_max = std::max(length_max, length_max_minus);
  output_info.write("PetscOperators: K2 parallel divergence loaded (conservation "
                    "{:.1e} forward, {:.1e} backward; length factor grid file / "
                    "mesh {:e} to {:e})\n",
                    error_plus, error_minus, length_min, length_max);
}

namespace {
/// Half-step leg fluxes K * Grad f on both sides with the boundary treatment
/// of the support operators (see getParallel): boundary legs weighted 2 if
/// bndry_flux, otherwise zero. Without K (nullptr) the flux is Grad f.
std::pair<PetscForwardLegVector, PetscBackwardLegVector>
legFluxes(const PetscOperators::Parallel& par, const Field3D* K, const Field3D& f,
          bool bndry_flux) {
  auto flux_plus = par.Grad_plus(f);
  auto flux_minus = par.Grad_minus(f);
  if (K != nullptr) {
    const auto K_plus = par.Interp_plus(*K);
    const auto K_minus = par.Interp_minus(*K);
    flux_plus = PetscForwardLegVector::pointwiseMultiply(K_plus, flux_plus);
    flux_minus = PetscBackwardLegVector::pointwiseMultiply(K_minus, flux_minus);
  }
  if (bndry_flux) {
    // A boundary face appears in only one half-operator (see getParallel):
    // weight 2 on boundary legs, i.e. flux * (2 - interior mask)
    flux_plus = (flux_plus * 2.0)
                - PetscForwardLegVector::pointwiseMultiply(flux_plus, par.Interior_plus);
    flux_minus =
        (flux_minus * 2.0)
        - PetscBackwardLegVector::pointwiseMultiply(flux_minus, par.Interior_minus);
  } else {
    // No flux through the parallel boundaries
    flux_plus = PetscForwardLegVector::pointwiseMultiply(flux_plus, par.Interior_plus);
    flux_minus =
        PetscBackwardLegVector::pointwiseMultiply(flux_minus, par.Interior_minus);
  }
  return {std::move(flux_plus), std::move(flux_minus)};
}

void requireK2(const PetscOperators::Parallel& par) {
  if (not par.hasK2()) {
    throw BoutException(
        "The K2 parallel divergence (conduction_method = PetscOperatorK2) needs the "
        "matrices k2_forward_rows/columns/weights and k2_backward_rows/columns/weights "
        "in the grid file, which this grid does not have. Regenerate the grid with a "
        "zoidberg version that writes them, or add them to the grid file.");
  }
}
} // namespace

Field3D PetscOperators::Parallel::Div_par_K_Grad_par(const Field3D& K, const Field3D& f,
                                                     bool bndry_flux) const {
  const auto [flux_plus, flux_minus] = legFluxes(*this, &K, f, bndry_flux);
  const auto div_minus = Div_minus(flux_plus);
  const auto div_plus = Div_plus(flux_minus);
  return (toField3D(div_minus, K) + toField3D(div_plus, K)) * 0.5;
}

Field3D PetscOperators::Parallel::Div_par_K_Grad_par_k2(const Field3D& K,
                                                        const Field3D& f,
                                                        bool bndry_flux) const {
  requireK2(*this);
  const auto [flux_plus, flux_minus] = legFluxes(*this, &K, f, bndry_flux);
  const auto div_minus = (*Div_minus_k2)(flux_plus);
  const auto div_plus = (*Div_plus_k2)(flux_minus);
  return (toField3D(div_minus, K) + toField3D(div_plus, K)) * 0.5;
}

Field3D PetscOperators::Parallel::Div_par_Grad_par_k2(const Field3D& f) const {
  requireK2(*this);
  const auto [flux_plus, flux_minus] = legFluxes(*this, nullptr, f, true);
  const auto div_minus = (*Div_minus_k2)(flux_plus);
  const auto div_plus = (*Div_plus_k2)(flux_minus);
  return (toField3D(div_minus, f) + toField3D(div_plus, f)) * 0.5;
}

PetscCellOperator makeNeumannOperator(const PetscCellMappingPtr& mapping,
                                      BoundaryDirection direction) {

  // The operator is built using stored indices for both rows and columns,
  // then post-multiplied by getPetscToStored() to convert columns to PETSc
  // ordering — exactly the same pattern as assembleFromCSR.

  const PetscInt n_global = mapping->globalSize();
  const PetscInt n_local = mapping->localSize();

  // Temporary matrix in stored-column ordering.
  // Each row has at most one nonzero (either diagonal or the interior
  // neighbour), so d_nz=1, o_nz=1 is the correct preallocation.
  bout::petsc::UniqueMat temp{new Mat{nullptr}};
  BOUT_DO_PETSC(MatCreate(BoutComm::get(), temp.get()));
  BOUT_DO_PETSC(MatSetType(*temp, MATMPIAIJ));
  BOUT_DO_PETSC(MatSetSizes(*temp, n_local, n_local, n_global, n_global));
  BOUT_DO_PETSC(MatMPIAIJSetPreallocation(*temp, 1, nullptr, 1, nullptr));

  const PetscScalar one = 1.0;

  // Iterate over all locally owned cells using mapLocalField (interior +
  // X-boundary) and mapLocalYup / mapLocalYdown (parallel boundary virtual
  // cells).  For each cell we insert exactly one entry.
  //
  // mapLocalField visits cells whose stored index comes from cell_number.
  // mapLocalYup visits cells whose stored index comes from forward_cell_number.
  // mapLocalYdown visits cells whose stored index comes from backward_cell_number.
  //
  // For a forward boundary virtual cell at field index i.yp():
  //   row = stored index from forward_cell_number (the virtual cell)
  //   col = stored index from cell_number at the source cell i
  //   => virtual cell copies from the adjacent interior cell (Neumann)
  //
  // For a backward boundary virtual cell at field index i.ym():
  //   row = stored index from backward_cell_number (the virtual cell)
  //   col = stored index from cell_number at the source cell i
  //
  // For all other cells the row is the diagonal (identity).

  // Identity rows: all cells in the interior and X-boundary regions
  mapping->mapLocalField([&](PetscInt local_row, const Ind3D& i) {
    const PetscInt petsc_row = mapping->rowStart() + local_row;
    const int stored_col = ROUND(mapping->cell_number[i]);
    if (stored_col < 0) {
      throw BoutException("Check failed: stored_col, mapLocalField");
    }
    const PetscInt col = static_cast<PetscInt>(stored_col);
    BOUT_DO_PETSC(MatSetValues(*temp, 1, &petsc_row, 1, &col, &one, INSERT_VALUES));
  });

  if ((direction == BoundaryDirection::Forward)
      or (direction == BoundaryDirection::Both)) {
    // Set forward rows from interior columns
    mapping->mapLocalYup([&](PetscInt local_row, const Ind3D& i_yp) {
      const PetscInt petsc_row = mapping->rowStart() + local_row;
      const Ind3D i_src = i_yp.ym();
      const int stored_row_check = ROUND(mapping->forward_cell_number[i_src]);
      if (stored_row_check < 0) {
        throw BoutException("Check failed: stored_row, mapLocalYup");
      }
      const int stored_col = ROUND(mapping->cell_number[i_src]);
      if (stored_col < 0) {
        throw BoutException("Check failed: stored_col, mapLocalYup");
      }
      const PetscInt col = static_cast<PetscInt>(stored_col);
      BOUT_DO_PETSC(MatSetValues(*temp, 1, &petsc_row, 1, &col, &one, INSERT_VALUES));
    });
  } else {
    // Identity rows in forward boundary
    mapping->mapLocalYup([&](PetscInt local_row, const Ind3D& i_yp) {
      const PetscInt petsc_row = mapping->rowStart() + local_row;
      const Ind3D i_src = i_yp.ym();
      const int stored_col = ROUND(mapping->forward_cell_number[i_src]);
      if (stored_col < 0) {
        throw BoutException("Check failed: stored_col, mapLocalYup identity");
      }
      const PetscInt col = static_cast<PetscInt>(stored_col);
      BOUT_DO_PETSC(MatSetValues(*temp, 1, &petsc_row, 1, &col, &one, INSERT_VALUES));
    });
  }

  if ((direction == BoundaryDirection::Backward)
      or (direction == BoundaryDirection::Both)) {
    // Set backward rows from interior columns
    mapping->mapLocalYdown([&](PetscInt local_row, const Ind3D& i_ym) {
      const PetscInt petsc_row = mapping->rowStart() + local_row;
      const Ind3D i_src = i_ym.yp();
      const int stored_row_check = ROUND(mapping->backward_cell_number[i_src]);
      if (stored_row_check < 0) {
        throw BoutException("Check failed: stored_row, mapLocalYdown");
      }
      const int stored_col = ROUND(mapping->cell_number[i_src]);
      if (stored_col < 0) {
        throw BoutException("Check failed: stored_col, mapLocalYdown");
      }
      const PetscInt col = static_cast<PetscInt>(stored_col);
      BOUT_DO_PETSC(MatSetValues(*temp, 1, &petsc_row, 1, &col, &one, INSERT_VALUES));
    });
  } else {
    // Identity rows in backward boundary
    mapping->mapLocalYdown([&](PetscInt local_row, const Ind3D& i_ym) {
      const PetscInt petsc_row = mapping->rowStart() + local_row;
      const Ind3D i_src = i_ym.yp();
      const int stored_col = ROUND(mapping->backward_cell_number[i_src]);
      if (stored_col < 0) {
        throw BoutException("Check failed: stored_col, mapLocalYdown identity");
      }
      const PetscInt col = static_cast<PetscInt>(stored_col);
      BOUT_DO_PETSC(MatSetValues(*temp, 1, &petsc_row, 1, &col, &one, INSERT_VALUES));
    });
  }

  BOUT_DO_PETSC(MatAssemblyBegin(*temp, MAT_FINAL_ASSEMBLY));
  BOUT_DO_PETSC(MatAssemblyEnd(*temp, MAT_FINAL_ASSEMBLY));

  // Post-multiply by getPetscToStored() to convert stored column indices to
  // PETSc column indices — identical to the final step in assembleFromCSR.
  bout::petsc::UniqueMat out{new Mat{nullptr}};
  BOUT_DO_PETSC(MatMatMult(*temp, mapping->getPetscToStored(), MAT_INITIAL_MATRIX,
                           PETSC_DETERMINE, out.get()));

  return PetscCellOperator(mapping, mapping, std::move(out));
}

Field3D PetscOperators::computeYup(const Field3D& f) const {
  ASSERT1(f.hasParallelSlices());
  ASSERT1(f.yup().isAllocated());
  ASSERT1(f.ydown().isAllocated());

  if (!cache_forward.has_value()) {
    cache_forward = this->forward();
  }
  auto leg = cache_forward.value()(f);

  const PetscScalar* leg_x{nullptr};
  BOUT_DO_PETSC(VecGetArrayRead(leg.raw(), &leg_x));

  // Points without an interior leg (field line leaves the domain) are
  // parallel boundary points, which boundary conditions set: leave them zero
  Field3D result{zeroFrom(f)};

  const PetscInt leg_row_start = forward_leg_mapping->rowStart();

  cell_mapping->mapOwnedInteriorCells([&](PetscInt /*cell_row*/, const Ind3D& i, int) {
    const int interior = ROUND(forward_leg_interior_number[i]);

    BoutReal value = 0.0;
    if (interior >= 0) {
      const PetscInt row = forward_leg_mapping->storedToPetsc(interior) - leg_row_start;
      value = leg_x[row];
    }
    result[i.yp()] = value;
  });

  BOUT_DO_PETSC(VecRestoreArrayRead(leg.raw(), &leg_x));

  return result;
}

Field3D PetscOperators::computeYdown(const Field3D& f) const {
  ASSERT1(f.hasParallelSlices());
  ASSERT1(f.yup().isAllocated());
  ASSERT1(f.ydown().isAllocated());

  if (!cache_backward.has_value()) {
    cache_backward = this->backward();
  }
  auto leg = cache_backward.value()(f);

  const PetscScalar* leg_x{nullptr};
  BOUT_DO_PETSC(VecGetArrayRead(leg.raw(), &leg_x));

  // Points without an interior leg (field line leaves the domain) are
  // parallel boundary points, which boundary conditions set: leave them zero
  Field3D result{zeroFrom(f)};

  const PetscInt leg_row_start = backward_leg_mapping->rowStart();

  cell_mapping->mapOwnedInteriorCells([&](PetscInt /*cell_row*/, const Ind3D& i, int) {
    const int interior = ROUND(backward_leg_interior_number[i]);

    BoutReal value = 0.0;
    if (interior >= 0) {
      const PetscInt row = backward_leg_mapping->storedToPetsc(interior) - leg_row_start;
      value = leg_x[row];
    }
    result[i.ym()] = value;
  });

  BOUT_DO_PETSC(VecRestoreArrayRead(leg.raw(), &leg_x));

  return result;
}

std::shared_ptr<PetscOperators> PetscOperators::Cache::get(Mesh* mesh) {
  if (auto search = petscOp.find(mesh); search != petscOp.end()) {
    return search->second;
  }
  auto op = std::make_shared<PetscOperators>(mesh);
  petscOp[mesh] = op;
  return op;
}

std::shared_ptr<PetscOperators::Parallel> PetscOperators::Cache::getParallel(Mesh* mesh) {
  if (auto search = parOp.find(mesh); search != parOp.end()) {
    return search->second;
  }
  auto op = get(mesh);
  auto par = std::make_shared<PetscOperators::Parallel>(op->getParallel());
  parOp[mesh] = par;
  return par;
}

void PetscOperators::Cache::release(Mesh* mesh) {
  // The Parallel operators are derived from the PetscOperators: release first
  parOp.erase(mesh);
  petscOp.erase(mesh);
}

void PetscOperators::Cache::cleanup() {
  parOp.clear();
  petscOp.clear();
}

std::map<Mesh*, std::shared_ptr<PetscOperators::Parallel>> PetscOperators::Cache::parOp;
std::map<Mesh*, std::shared_ptr<PetscOperators>> PetscOperators::Cache::petscOp;
#endif
