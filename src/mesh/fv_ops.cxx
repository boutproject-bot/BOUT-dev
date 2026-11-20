#include "bout/fv_ops.hxx"
#include "bout/fv_ops_impl.hxx"

#include "bout/assert.hxx"
#include "bout/bout_types.hxx"
#include "bout/boutexception.hxx"
#include "bout/build_config.hxx"
#include "bout/coordinates.hxx"
#include "bout/field2d.hxx"
#include "bout/field3d.hxx"
#include "bout/globals.hxx"
#include "bout/msg_stack.hxx"
#include "bout/output.hxx"
#include "bout/region.hxx"
#include "bout/utils.hxx"
#include <bout/yboundary_regions.hxx>
#if BOUT_HAS_PETSC
#include "bout/petsc_operators.hxx"
#include <petscvec.h>
#endif

#include <algorithm>
#include <limits>
#include <map>
#include <math.h>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace {
template <class T>
struct Slices {
  T c;
  T up;
  T down;
  Slices(bool use_slices, const T& field)
      : c(use_slices ? field : toFieldAligned(field)), up(use_slices ? field.yup() : c),
        down(use_slices ? field.ydown() : c) {}
};
template <class T>
Slices<T> makeslices(bool use_slices, const T& field) {
  return Slices<T>(use_slices, field);
}
} // namespace

namespace FV {

Field3D Div_a_Grad_perp(const Field3D& a, const Field3D& f) {
  ASSERT2(a.getLocation() == f.getLocation());

  Mesh* mesh = a.getMesh();

  Field3D result{zeroFrom(f)};

  Coordinates* coord = f.getCoordinates();

  // Flux in x

  const int xs = mesh->xstart - 1;
  const int xe = mesh->xend;
  if (xs >= 0) {
    for (int i = xs; i <= xe; i++) {
      for (int j = mesh->ystart; j <= mesh->yend; j++) {
        for (int k = mesh->zstart; k <= mesh->zend; k++) {
          // Calculate flux from i to i+1

          const BoutReal fout = 0.5 * (a(i, j, k) + a(i + 1, j, k))
                                * (coord->J(i, j, k) * coord->g11(i, j, k)
                                   + coord->J(i + 1, j, k) * coord->g11(i + 1, j, k))
                                * (f(i + 1, j, k) - f(i, j, k))
                                / (coord->dx(i, j, k) + coord->dx(i + 1, j, k));

          result(i, j, k) += fout / (coord->dx(i, j, k) * coord->J(i, j, k));
          result(i + 1, j, k) -= fout / (coord->dx(i + 1, j, k) * coord->J(i + 1, j, k));
        }
      }
    }
  } else {
    ASSERT1(mesh->xend == 0);
  }

  const bool fci = f.hasParallelSlices() && a.hasParallelSlices();

  if (bout::build::use_metric_3d and fci) {
    // 3D Metric, need yup/ydown fields.
    // Requires previous communication of metrics
    // -- should insert communication here?
    if (!coord->g23().hasParallelSlices() || !coord->g_23().hasParallelSlices()
        || !coord->dy().hasParallelSlices() || !coord->dz().hasParallelSlices()
        || !coord->Bxy().hasParallelSlices() || !coord->J().hasParallelSlices()) {
      throw BoutException("metrics have no yup/down!");
    }
  }

  // A singleton global Y mesh has no Y derivatives, even if guards exist.
  const bool y_derivatives = mesh->GlobalNy > 1;
  ASSERT1(!y_derivatives || mesh->ystart > 0);

  // Y fluxes and the mixed part of the Z flux require Y derivatives.

  // Fields containing values along the magnetic field

  // Values on this y slice (centre).
  // This is needed because toFieldAligned may modify the field
  const auto f_slice = makeslices(fci, f);
  const auto a_slice = makeslices(fci, a);

  // Only in 3D case with FCI do the metrics have parallel slices
  const bool metric_fci = fci and bout::build::use_metric_3d;
  const auto g23 = makeslices(metric_fci, coord->g23());
  const auto g_23 = makeslices(metric_fci, coord->g_23());
  const auto J = makeslices(metric_fci, coord->J());
  const auto dy = makeslices(metric_fci, coord->dy());
  const auto dz = makeslices(metric_fci, coord->dz());
  const auto Bxy = makeslices(metric_fci, coord->Bxy());

  // Result of the Y and Z fluxes
  Field3D yzresult(0.0, mesh);
  if (!fci) {
    yzresult.setDirectionY(YDirectionType::Aligned);
  }

  // Y flux
  if (y_derivatives) {
    BOUT_FOR(i, yzresult.getRegion("RGN_NOBNDRY")) {
      auto ikp = i.zp();
      auto ikm = i.zm();
      {
        const BoutReal coef = 0.5
                              * (g_23.c[i] / SQ(J.c[i] * Bxy.c[i])
                                 + g_23.up[i.yp()] / SQ(J.up[i.yp()] * Bxy.up[i.yp()]));

        // Calculate Z derivative at y boundary
        const BoutReal dfdz = 0.5
                              * (f_slice.c[ikp] - f_slice.c[ikm] + f_slice.up[ikp.yp()]
                                 - f_slice.up[ikm.yp()])
                              / (dz.c[i] + dz.up[i.yp()]);

        // Y derivative
        const BoutReal dfdy =
            2. * (f_slice.up[i.yp()] - f_slice.c[i]) / (dy.up[i.yp()] + dy.c[i]);

        const BoutReal fout = 0.25 * (a_slice.c[i] + a_slice.up[i.yp()])
                              * (J.c[i] * g23.c[i] + J.up[i.yp()] * g23.up[i.yp()])
                              * (dfdz - coef * dfdy);

        yzresult[i] += fout / (dy.c[i] * J.c[i]);
      }
      {
        // Calculate flux between j and j-1
        const BoutReal coef =
            0.5
            * (g_23.c[i] / SQ(J.c[i] * Bxy.c[i])
               + g_23.down[i.ym()] / SQ(J.down[i.ym()] * Bxy.down[i.ym()]));

        const BoutReal dfdz = 0.5
                              * (f_slice.c[ikp] - f_slice.c[ikm] + f_slice.down[ikp.ym()]
                                 - f_slice.down[ikm.ym()])
                              / (dz.c[i] + dz.down[i.ym()]);

        const BoutReal dfdy =
            2. * (f_slice.c[i] - f_slice.down[i.ym()]) / (dy.c[i] + dy.down[i.ym()]);

        const BoutReal fout = 0.25 * (a_slice.c[i] + a_slice.down[i.ym()])
                              * (J.c[i] * g23.c[i] + J.down[i.ym()] * g23.down[i.ym()])
                              * (dfdz - coef * dfdy);

        yzresult[i] -= fout / (dy.c[i] * J.c[i]);
      }
    }
  }

  // Z flux exists independently of Y guards or Y derivatives.
  BOUT_FOR(i, yzresult.getRegion("RGN_NOBNDRY")) {
    const auto ikp = i.zp();
    BoutReal y_correction = 0.0;
    if (y_derivatives) {
      const BoutReal coef = g_23.c[i] / (dy.up[i.yp()] + 2. * dy.c[i] + dy.down[i.ym()])
                            / SQ(J.c[i] * Bxy.c[i]);
      y_correction = coef
                     * (f_slice.up[i.yp()] + f_slice.up[ikp.yp()] - f_slice.down[i.ym()]
                        - f_slice.down[ikp.ym()]);
    }

    const BoutReal fout = 0.25 * (a_slice.c[i] + a_slice.c[ikp])
                          * (J.c[i] * coord->g33()[i] + J.c[ikp] * coord->g33()[ikp])
                          * ((f_slice.c[ikp] - f_slice.c[i]) / dz.c[i] - y_correction);

    yzresult[i] += fout / (J.c[i] * dz.c[i]);
    yzresult[ikp] -= fout / (J.c[ikp] * dz.c[ikp]);
  }

  // Check if we need to transform back
  if (fci) {
    result += yzresult;
  } else {
    result += fromFieldAligned(yzresult);
  }

  return result;
}

Field3D Div_par_K_Grad_par(const Field3DParallel& Kin, const Field3DParallel& fin,
                           bool bndry_flux) {

  ASSERT2(Kin.getLocation() == fin.getLocation());

  const Mesh* mesh = Kin.getMesh();

  const bool use_parallel_slices = (Kin.hasParallelSlices() && fin.hasParallelSlices());

  const auto& K = use_parallel_slices ? Kin : toFieldAligned(Kin, "RGN_NOX");
  const auto& f = use_parallel_slices ? fin : toFieldAligned(fin, "RGN_NOX");

  Field3D result{zeroFrom(f)};

  // K and f fields in yup and ydown directions
  const auto& Kup = use_parallel_slices ? Kin.yup() : K;
  const auto& Kdown = use_parallel_slices ? Kin.ydown() : K;
  const auto& fup = use_parallel_slices ? fin.yup() : f;
  const auto& fdown = use_parallel_slices ? fin.ydown() : f;

  const Coordinates* coord = fin.getCoordinates();
  const auto& yboundary = coord->getYBoundary();

  BOUT_FOR(i, result.getRegion("RGN_NOBNDRY")) {
    // Calculate flux at upper surface

    if (bndry_flux
        || (not K.isFci()
            and (mesh->periodicY(i.x()) || !mesh->lastY(i.x()) || (i.y() != mesh->yend)))
        or (K.isFci() and yboundary.contains<+1>(i))) {
      const auto iyp = i.yp();
      const BoutReal c = 0.5 * (K[i] + Kup[iyp]); // K at the upper boundary
      const BoutReal J = 0.5 * (coord->J()[i] + coord->J()[iyp]); // Jacobian at boundary
      const BoutReal g_22 = 0.5 * (coord->g_22()[i] + coord->g_22()[iyp]);

      const BoutReal gradient =
          2. * (fup[iyp] - f[i]) / (coord->dy()[i] + coord->dy()[iyp]);

      const BoutReal flux = c * J * gradient / g_22;

      result[i] += flux / (coord->dy()[i] * coord->J()[i]);
    }

    // Calculate flux at lower surface
    if (bndry_flux
        || (not K.isFci()
            and (mesh->periodicY(i.x()) || !mesh->firstY(i.x())
                 || (i.y() != mesh->ystart)))
        or (K.isFci() and yboundary.contains<-1>(i))) {
      const auto iym = i.ym();
      const BoutReal c = 0.5 * (K[i] + Kdown[iym]); // K at the lower boundary
      const BoutReal J = 0.5 * (coord->J()[i] + coord->J()[iym]); // Jacobian at boundary

      const BoutReal g_22 = 0.5 * (coord->g_22()[i] + coord->g_22()[iym]);

      const BoutReal gradient =
          2. * (f[i] - fdown[iym]) / (coord->dy()[i] + coord->dy()[iym]);

      const BoutReal flux = c * J * gradient / g_22;

      result[i] -= flux / (coord->dy()[i] * coord->J()[i]);
    }
  }

  if (!use_parallel_slices) {
    // Shifted to field aligned coordinates, so need to shift back
    result = fromFieldAligned(result, "RGN_NOBNDRY");
  }

  return result;
}

Field3D D4DY4(const Field3D& d_in, const Field3D& f_in) {
  ASSERT1_FIELDS_COMPATIBLE(d_in, f_in);

  const Mesh* mesh = d_in.getMesh();

  Coordinates* coord = f_in.getCoordinates();

  ASSERT2(d_in.getDirectionY() == f_in.getDirectionY());
  const bool are_unaligned = ((d_in.getDirectionY() == YDirectionType::Standard)
                              and (f_in.getDirectionY() == YDirectionType::Standard));

  // Convert to field aligned coordinates
  Field3D d = are_unaligned ? toFieldAligned(d_in, "RGN_NOX") : d_in;
  Field3D f = are_unaligned ? toFieldAligned(f_in, "RGN_NOX") : f_in;

  Field3D result{zeroFrom(f)};

  for (int i = mesh->xstart; i <= mesh->xend; i++) {
    // Check for boundaries
    const bool yperiodic = mesh->periodicY(i);
    const bool has_upper_boundary = !yperiodic && mesh->lastY(i);
    const bool has_lower_boundary = !yperiodic && mesh->firstY(i);

    // Always calculate fluxes at upper Y cell boundary
    const int ystart =
        has_lower_boundary
            ? mesh->ystart
            : // Don't calculate flux from boundary mesh->ystart-1 into domain
            mesh->ystart - 1; // Calculate flux from last guard cell into domain

    const int yend = has_upper_boundary
                         ? mesh->yend - 1
                         : // Don't calculate flux from mesh->yend into boundary
                         mesh->yend;

    for (int j = ystart; j <= yend; j++) {
      for (int k = mesh->zstart; k <= mesh->zend; k++) {
        const BoutReal dy3 = SQ(coord->dy(i, j, k)) * coord->dy(i, j, k);
        // 3rd derivative at upper boundary

        const BoutReal d3fdy3 =
            (f(i, j + 2, k) - 3. * f(i, j + 1, k) + 3. * f(i, j, k) - f(i, j - 1, k))
            / dy3;

        const BoutReal flux = 0.5 * (d(i, j, k) + d(i, j + 1, k))
                              * (coord->J(i, j, k) + coord->J(i, j + 1, k)) * d3fdy3;

        result(i, j, k) += flux / (coord->J(i, j, k) * coord->dy(i, j, k));
        result(i, j + 1, k) -= flux / (coord->J(i, j + 1, k) * coord->dy(i, j + 1, k));
      }
    }
  }

  // Convert result back to non-aligned coordinates
  return are_unaligned ? fromFieldAligned(result, "RGN_NOBNDRY") : result;
}

Field3D D4DY4_Index(const Field3D& f_in, bool bndry_flux) {
  const Mesh* mesh = f_in.getMesh();

  // Convert to field aligned coordinates
  const bool is_unaligned = (f_in.getDirectionY() == YDirectionType::Standard);
  Field3D f = is_unaligned ? toFieldAligned(f_in, "RGN_NOX") : f_in;

  Field3D result{zeroFrom(f)};

  Coordinates* coord = f_in.getCoordinates();

  for (int i = mesh->xstart; i <= mesh->xend; i++) {
    const bool yperiodic = mesh->periodicY(i);

    const bool has_upper_boundary = !yperiodic && mesh->lastY(i);
    const bool has_lower_boundary = !yperiodic && mesh->firstY(i);

    for (int j = mesh->ystart; j <= mesh->yend; j++) {

      // Right boundary

      if (bndry_flux || (j != mesh->yend) || !has_upper_boundary) {
        // Calculate the fluxes

        if (j != mesh->yend || !has_upper_boundary) {

          for (int k = mesh->zstart; k <= mesh->zend; k++) {
            // Right boundary common factors
            const BoutReal common_factor = 0.25
                                           * (coord->dy(i, j, k) + coord->dy(i, j + 1, k))
                                           * (coord->J(i, j, j) + coord->J(i, j + 1, k));

            const BoutReal factor_rc =
                common_factor / (coord->J(i, j, k) * coord->dy(i, j, k));
            const BoutReal factor_rp =
                common_factor / (coord->J(i, j + 1, k) * coord->dy(i, j + 1, k));

            // Not on domain boundary
            // 3rd derivative at right cell boundary

            const BoutReal d3fdx3 = (f(i, j + 2, k) - (3. * f(i, j + 1, k))
                                     + (3. * f(i, j, k)) - f(i, j - 1, k));

            result(i, j, k) += d3fdx3 * factor_rc;
            result(i, j + 1, k) -= d3fdx3 * factor_rp;
          }
        } else {
          // At a domain boundary
          // Use a one-sided difference formula

          for (int k = mesh->zstart; k <= mesh->zend; k++) {
            // Right boundary common factors
            const BoutReal common_factor = 0.25
                                           * (coord->dy(i, j, k) + coord->dy(i, j + 1, k))
                                           * (coord->J(i, j, j) + coord->J(i, j + 1, k));

            const BoutReal factor_rc =
                common_factor / (coord->J(i, j, k) * coord->dy(i, j, k));
            const BoutReal factor_rp =
                common_factor / (coord->J(i, j + 1, k) * coord->dy(i, j + 1, k));

            const BoutReal d3fdx3 =
                -(((16. / 5) * 0.5 * (f(i, j + 1, k) + f(i, j, k))) // Boundary value f_b
                  - (6. * f(i, j, k))                               // f_0
                  + (4. * f(i, j - 1, k))                           // f_1
                  - ((6. / 5) * f(i, j - 2, k))                     // f_2
                );

            result(i, j, k) += d3fdx3 * factor_rc;
            result(i, j + 1, k) -= d3fdx3 * factor_rp;
          }
        }
      }

      // Left cell boundary

      if (bndry_flux || (j != mesh->ystart) || !has_lower_boundary) {
        // Calculate the fluxes

        if (j != mesh->ystart || !has_lower_boundary) {
          for (int k = mesh->zstart; k <= mesh->zend; k++) {
            const BoutReal common_factor = 0.25
                                           * (coord->dy(i, j, k) + coord->dy(i, j + 1, k))
                                           * (coord->J(i, j, k) + coord->J(i, j - 1, k));

            const BoutReal factor_lc =
                common_factor / (coord->J(i, j, k) * coord->dy(i, j, k));
            const BoutReal factor_lm =
                common_factor / (coord->J(i, j - 1, k) * coord->dy(i, j - 1, k));

            // Not on a domain boundary
            const BoutReal d3fdx3 = (f(i, j + 1, k) - (3. * f(i, j, k))
                                     + (3. * f(i, j - 1, k)) - f(i, j - 2, k));

            result(i, j, k) -= d3fdx3 * factor_lc;
            result(i, j - 1, k) += d3fdx3 * factor_lm;
          }
        } else {
          // On a domain (Y) boundary
          for (int k = mesh->zstart; k <= mesh->zend; k++) {
            const BoutReal common_factor = 0.25
                                           * (coord->dy(i, j, k) + coord->dy(i, j + 1, k))
                                           * (coord->J(i, j, k) + coord->J(i, j - 1, k));

            const BoutReal factor_lc =
                common_factor / (coord->J(i, j, k) * coord->dy(i, j, k));
            const BoutReal factor_lm =
                common_factor / (coord->J(i, j - 1, k) * coord->dy(i, j - 1, k));
            const BoutReal d3fdx3 =
                -((-(16. / 5) * 0.5 * (f(i, j - 1, k) + f(i, j, k))) // Boundary value f_b
                  + (6. * f(i, j, k))                                // f_0
                  - (4. * f(i, j + 1, k))                            // f_1
                  + ((6. / 5) * f(i, j + 2, k))                      // f_2
                );

            result(i, j, k) -= d3fdx3 * factor_lc;
            result(i, j - 1, k) += d3fdx3 * factor_lm;
          }
        }
      }
    }
  }

  // Convert result back to non-aligned coordinates
  return is_unaligned ? fromFieldAligned(result, "RGN_NOBNDRY") : result;
}

void communicateFluxes(Field3D& f) {
  Mesh* mesh = f.getMesh();

  // Use X=0 as temporary buffer
  if (mesh->xstart != 2) {
    throw BoutException("communicateFluxes: Sorry!");
  }

  const int size = mesh->LocalNy * mesh->LocalNz;
  comm_handle xin = nullptr;
  comm_handle xout = nullptr;
  // Cache results to silence spurious compiler warning about xin,
  // xout possibly being uninitialised when used
  const bool not_first = mesh->periodicX || !mesh->firstX();
  const bool not_last = mesh->periodicX || !mesh->lastX();

  if (not_first) {
    xin = mesh->irecvXIn(f(0, 0), size, 0);
  }
  if (not_last) {
    xout = mesh->irecvXOut(f(mesh->LocalNx - 1, 0), size, 1);
  }
  // Send X=1 values
  if (not_first) {
    mesh->sendXIn(f(1, 0), size, 1);
  }
  if (not_last) {
    mesh->sendXOut(f(mesh->LocalNx - 2, 0), size, 0);
  }
  // Wait
  if (not_first) {
    mesh->wait(xin);
    // Add to cells
    for (int y = mesh->ystart; y <= mesh->yend; y++) {
      for (int z = mesh->zstart; z <= mesh->zend; z++) {
        f(2, y, z) += f(0, y, z);
      }
    }
  }
  if (not_last) {
    mesh->wait(xout);
    // Add to cells
    for (int y = mesh->ystart; y <= mesh->yend; y++) {
      for (int z = mesh->zstart; z <= mesh->zend; z++) {
        f(mesh->LocalNx - 3, y, z) += f(mesh->LocalNx - 1, y, z);
      }
    }
  }
}

Field3D Div_Perp_Lap(const Field3D& a, const Field3D& f, CELL_LOC outloc) {

  Field3D result = 0.0;

  //////////////////////////////////////////
  // X-Z diffusion
  //
  //            Z
  //            |
  //
  //     o --- gU --- o
  //     |     nU     |
  //     |            |
  //    gL nL      nR gR    -> X
  //     |            |
  //     |     nD     |
  //     o --- gD --- o
  //
  Coordinates* coords = a.getCoordinates(outloc);
  const Mesh* mesh = f.getMesh();

  for (int i = mesh->xstart; i <= mesh->xend; i++) {
    for (int j = mesh->ystart; j <= mesh->yend; j++) {
      for (int k = 0; k < mesh->LocalNz; k++) {

        // wrap k-index around as Z is (currently) periodic.
        const int kp = (k + 1) % (mesh->LocalNz);
        const int km = (k - 1 + mesh->LocalNz) % (mesh->LocalNz);

        // Calculate gradients on cell faces -- assumes constant grid spacing

        const BoutReal gR =
            ((coords->g11(i, j, k) + coords->g11(i + 1, j, k))
             * (f(i + 1, j, k) - f(i, j, k))
             / (coords->dx(i + 1, j, k) + coords->dx(i, j, k)))
            + (0.5 * (coords->g13(i, j, k) + coords->g13(i + 1, j, k))
               * (f(i + 1, j, kp) - f(i + 1, j, km) + f(i, j, kp) - f(i, j, km))
               / (4. * coords->dz(i, j, k)));

        const BoutReal gL =
            ((coords->g11(i - 1, j, k) + coords->g11(i, j, k))
             * (f(i, j, k) - f(i - 1, j, k))
             / (coords->dx(i - 1, j, k) + coords->dx(i, j, k)))
            + (0.5 * (coords->g13(i - 1, j, k) + coords->g13(i, j, k))
               * (f(i - 1, j, kp) - f(i - 1, j, km) + f(i, j, kp) - f(i, j, km))
               / (4 * coords->dz(i, j, k)));

        const BoutReal gD =
            (coords->g13(i, j, k)
             * (f(i + 1, j, km) - f(i - 1, j, km) + f(i + 1, j, k) - f(i - 1, j, k))
             / (4. * coords->dx(i, j, k)))
            + (coords->g33(i, j, k) * (f(i, j, k) - f(i, j, km)) / coords->dz(i, j, k));

        const BoutReal gU =
            (coords->g13(i, j, k)
             * (f(i + 1, j, kp) - f(i - 1, j, kp) + f(i + 1, j, k) - f(i - 1, j, k))
             / (4. * coords->dx(i, j, k)))
            + (coords->g33(i, j, k) * (f(i, j, kp) - f(i, j, k)) / coords->dz(i, j, k));

        // Flow right
        BoutReal flux = gR * 0.25 * (coords->J(i + 1, j, k) + coords->J(i, j, k))
                        * (a(i + 1, j, k) + a(i, j, k));
        result(i, j, k) += flux / (coords->dx(i, j, k) * coords->J(i, j, k));

        // Flow left
        flux = gL * 0.25 * (coords->J(i - 1, j, k) + coords->J(i, j, k))
               * (a(i - 1, j, k) + a(i, j, k));
        result(i, j, k) -= flux / (coords->dx(i, j, k) * coords->J(i, j, k));

        // Flow up
        flux = gU * 0.25 * (coords->J(i, j, k) + coords->J(i, j, kp))
               * (a(i, j, k) + a(i, j, kp));
        result(i, j, k) += flux / (coords->dz(i, j, k) * coords->J(i, j, k));

        // Flow down
        flux = gD * 0.25 * (coords->J(i, j, km) + coords->J(i, j, k))
               * (a(i, j, km) + a(i, j, k));
        result(i, j, k) += flux / (coords->dz(i, j, k) * coords->J(i, j, k));
      }
    }
  }

  return result;
}

// Explicit instantiations of flux-limited finite volume methods
template Field3D Div_par_fvv<Upwind>(const Field3D& f_in, const Field3D& v_in,
                                     const Field3D& wave_speed_in, bool fixflux);
template Field3D Div_par<Upwind>(const Field3D& f_in, const Field3D& v_in,
                                 const Field3D& wave_speed_in, bool fixflux);
template Field3D Div_f_v<Upwind>(const Field3D& n_in, const Vector3D& v, bool bndry_flux);
template Field3D Div_par_mod<Upwind>(const Field3D& f_in, const Field3D& v_in,
                                     const Field3D& wave_speed_in, Field3D& flow_ylow,
                                     bool fixflux, bool dissipative);
template Field3D Div_par_fvv_heating<Upwind>(const Field3D& f_in, const Field3D& v_in,
                                             const Field3D& wave_speed_in,
                                             Field3D& flow_ylow, bool fixflux);
template Field3D Div_a_Grad_perp_limit<Upwind>(const Field3D& a, const Field3D& g,
                                               const Field3D& f);

template Field3D Div_par_fvv<Fromm>(const Field3D& f_in, const Field3D& v_in,
                                    const Field3D& wave_speed_in, bool fixflux);
template Field3D Div_par<Fromm>(const Field3D& f_in, const Field3D& v_in,
                                const Field3D& wave_speed_in, bool fixflux);
template Field3D Div_f_v<Fromm>(const Field3D& n_in, const Vector3D& v, bool bndry_flux);
template Field3D Div_par_mod<Fromm>(const Field3D& f_in, const Field3D& v_in,
                                    const Field3D& wave_speed_in, Field3D& flow_ylow,
                                    bool fixflux, bool dissipative);
template Field3D Div_par_fvv_heating<Fromm>(const Field3D& f_in, const Field3D& v_in,
                                            const Field3D& wave_speed_in,
                                            Field3D& flow_ylow, bool fixflux);
template Field3D Div_a_Grad_perp_limit<Fromm>(const Field3D& a, const Field3D& g,
                                              const Field3D& f);

template Field3D Div_par_fvv<MinMod>(const Field3D& f_in, const Field3D& v_in,
                                     const Field3D& wave_speed_in, bool fixflux);
template Field3D Div_par<MinMod>(const Field3D& f_in, const Field3D& v_in,
                                 const Field3D& wave_speed_in, bool fixflux);
template Field3D Div_f_v<MinMod>(const Field3D& n_in, const Vector3D& v, bool bndry_flux);
template Field3D Div_par_mod<MinMod>(const Field3D& f_in, const Field3D& v_in,
                                     const Field3D& wave_speed_in, Field3D& flow_ylow,
                                     bool fixflux, bool dissipative);
template Field3D Div_par_fvv_heating<MinMod>(const Field3D& f_in, const Field3D& v_in,
                                             const Field3D& wave_speed_in,
                                             Field3D& flow_ylow, bool fixflux);
template Field3D Div_a_Grad_perp_limit<MinMod>(const Field3D& a, const Field3D& g,
                                               const Field3D& f);

template Field3D Div_par_fvv<MC>(const Field3D& f_in, const Field3D& v_in,
                                 const Field3D& wave_speed_in, bool fixflux);
template Field3D Div_par<MC>(const Field3D& f_in, const Field3D& v_in,
                             const Field3D& wave_speed_in, bool fixflux);
template Field3D Div_f_v<MC>(const Field3D& n_in, const Vector3D& v, bool bndry_flux);
template Field3D Div_par_mod<MC>(const Field3D& f_in, const Field3D& v_in,
                                 const Field3D& wave_speed_in, Field3D& flow_ylow,
                                 bool fixflux, bool dissipative);
template Field3D Div_par_fvv_heating<MC>(const Field3D& f_in, const Field3D& v_in,
                                         const Field3D& wave_speed_in, Field3D& flow_ylow,
                                         bool fixflux);
template Field3D Div_a_Grad_perp_limit<MC>(const Field3D& a, const Field3D& g,
                                           const Field3D& f);

template Field3D Div_par_fvv<Superbee>(const Field3D& f_in, const Field3D& v_in,
                                       const Field3D& wave_speed_in, bool fixflux);
template Field3D Div_par<Superbee>(const Field3D& f_in, const Field3D& v_in,
                                   const Field3D& wave_speed_in, bool fixflux);
template Field3D Div_f_v<Superbee>(const Field3D& n_in, const Vector3D& v,
                                   bool bndry_flux);
template Field3D Div_par_mod<Superbee>(const Field3D& f_in, const Field3D& v_in,
                                       const Field3D& wave_speed_in, Field3D& flow_ylow,
                                       bool fixflux, bool dissipative);
template Field3D Div_par_fvv_heating<Superbee>(const Field3D& f_in, const Field3D& v_in,
                                               const Field3D& wave_speed_in,
                                               Field3D& flow_ylow, bool fixflux);
template Field3D Div_a_Grad_perp_limit<Superbee>(const Field3D& a, const Field3D& g,
                                                 const Field3D& f);

template Field3D Div_par_fvv<VanAlbada>(const Field3D& f_in, const Field3D& v_in,
                                        const Field3D& wave_speed_in, bool fixflux);
template Field3D Div_par<VanAlbada>(const Field3D& f_in, const Field3D& v_in,
                                    const Field3D& wave_speed_in, bool fixflux);
template Field3D Div_f_v<VanAlbada>(const Field3D& n_in, const Vector3D& v,
                                    bool bndry_flux);
template Field3D Div_par_mod<VanAlbada>(const Field3D& f_in, const Field3D& v_in,
                                        const Field3D& wave_speed_in, Field3D& flow_ylow,
                                        bool fixflux, bool dissipative);
template Field3D Div_par_fvv_heating<VanAlbada>(const Field3D& f_in, const Field3D& v_in,
                                                const Field3D& wave_speed_in,
                                                Field3D& flow_ylow, bool fixflux);
template Field3D Div_a_Grad_perp_limit<VanAlbada>(const Field3D& a, const Field3D& g,
                                                  const Field3D& f);

template Field3D Div_par_fvv<WENO3>(const Field3D& f_in, const Field3D& v_in,
                                    const Field3D& wave_speed_in, bool fixflux);
template Field3D Div_par<WENO3>(const Field3D& f_in, const Field3D& v_in,
                                const Field3D& wave_speed_in, bool fixflux);
template Field3D Div_f_v<WENO3>(const Field3D& n_in, const Vector3D& v, bool bndry_flux);
template Field3D Div_par_mod<WENO3>(const Field3D& f_in, const Field3D& v_in,
                                    const Field3D& wave_speed_in, Field3D& flow_ylow,
                                    bool fixflux, bool dissipative);
template Field3D Div_par_fvv_heating<WENO3>(const Field3D& f_in, const Field3D& v_in,
                                            const Field3D& wave_speed_in,
                                            Field3D& flow_ylow, bool fixflux);
template Field3D Div_a_Grad_perp_limit<WENO3>(const Field3D& a, const Field3D& g,
                                              const Field3D& f);

std::map<Mesh*, std::weak_ptr<Div_a_Grad_perp_FCI>>
    Div_a_Grad_perp_FCI_structured::op_cache;
std::map<Mesh*, std::weak_ptr<Div_a_Grad_perp_FCI>>
    Div_a_Grad_perp_FCI_unstructured::op_cache;
std::shared_ptr<Div_a_Grad_perp_FCI>
Div_a_Grad_perp_FCI_structured::create(Mesh* mesh, BoutReal rho_s0) {
  if (auto search = Div_a_Grad_perp_FCI_structured::op_cache.find(mesh);
      search != Div_a_Grad_perp_FCI_structured::op_cache.end()) {
    if (std::shared_ptr<Div_a_Grad_perp_FCI> spt = search->second.lock()) {
      return spt;
    }
    output_warn.write("\tPreformance Warning: DIV_A_GRAD_PERP needs to be re-created!\n");
  }
  auto dagp = std::make_shared<Div_a_Grad_perp_FCI_structured>(mesh);
  dagp->volume /= rho_s0 * rho_s0;
  Div_a_Grad_perp_FCI_structured::op_cache[mesh] = dagp;
  return dagp;
}

Div_a_Grad_perp_FCI_structured::Div_a_Grad_perp_FCI_structured(Mesh* mesh)
    : fac_XX(mesh), fac_XZ(mesh), fac_ZX(mesh), fac_ZZ(mesh), volume(mesh) {
  ASSERT0(mesh->get(fac_XX, "dagp_fv_XX", 0.0, false) == 0);
  ASSERT0(mesh->get(fac_XZ, "dagp_fv_XZ", 0.0, false) == 0);
  ASSERT0(mesh->get(fac_ZX, "dagp_fv_ZX", 0.0, false) == 0);
  ASSERT0(mesh->get(fac_ZZ, "dagp_fv_ZZ", 0.0, false) == 0);
  ASSERT0(volume.hasParallelSlices() == false);
  ASSERT0(mesh->get(volume, "dagp_fv_volume", 0.0, false) == 0);
  ASSERT0(volume.hasParallelSlices() == false);
  volume.setRegion("RGN_NOBNDRY");
  if (!mesh->hasRegion3D("RGN_dapg_fv_xbndry")) {
    mesh->addRegion("RGN_dapg_fv_xbndry",
                    Region<>(mesh->xstart - 1, mesh->xstart - 1, mesh->ystart, mesh->yend,
                             mesh->zstart, mesh->zend, mesh->LocalNy, mesh->LocalNz));
  }
}

Field3D Div_a_Grad_perp_FCI_structured::operator()(const Field3D& a, const Field3D& f,
                                                   Field3D& flow_xlow, Field3D& flow_zlow,
                                                   bool upwinding) const {
  if (upwinding) {
    return operator()<true, true>(a, f, &flow_xlow, &flow_zlow);
  }
  return operator()<true, false>(a, f, &flow_xlow, &flow_zlow);
}

Field3D Div_a_Grad_perp_FCI_structured::operator()(const Field3D& a, const Field3D& f,
                                                   bool upwinding) const {
  if (upwinding) {
    return operator()<false, true>(a, f, nullptr, nullptr);
  }
  return operator()<false, false>(a, f, nullptr, nullptr);
}

template <bool extra, bool upwinding>
Field3D Div_a_Grad_perp_FCI_structured::operator()(const Field3D& a, const Field3D& f,
                                                   Field3D* flow_xlow,
                                                   Field3D* flow_zlow) const {
  auto result{zeroFrom(f)};
  ASSERT1_FIELDS_COMPATIBLE(a, f);
#if BOUT_HAS_OPENMP
  bool parallel = omp_get_max_threads() > 1;
#else
  bool parallel = false;
#endif
  if constexpr (extra) {
    flow_xlow->allocate();
    flow_zlow->allocate();
  }
  Field3D xtmp, ztmp;
  if (parallel and not extra) {
    xtmp = emptyFrom(f);
    flow_xlow = &xtmp;
    ztmp = emptyFrom(f);
    flow_zlow = &ztmp;
  }
  if (not parallel) {
    BOUT_FOR(i, f.getRegion("RGN_dapg_fv_xbndry")) {
      const auto xf = xflux<upwinding>(a, f, i);
      result[i.xp()] = xf;
      if constexpr (extra) {
        (*flow_xlow)[i.xp()] = xf;
      }
    }
    BOUT_FOR(i, f.getRegion("RGN_NOBNDRY")) {
      const auto xf = xflux<upwinding>(a, f, i);
      result[i.xp()] = xf;
      result[i] -= xf;
      if constexpr (extra) {
        (*flow_xlow)[i.xp()] = xf;
      }
      const auto zf = zflux<upwinding>(a, f, i);
      result[i.zp()] += zf;
      result[i] -= zf;
      if constexpr (extra) {
        (*flow_zlow)[i.zp()] = zf;
      }
    }
  } else {
    BOUT_FOR(i, f.getRegion("RGN_dapg_fv_xbndry")) {
      (*flow_xlow)[i] = xflux<upwinding>(a, f, i.xm());
      ;
    }
    BOUT_FOR(i, f.getRegion("RGN_NOBNDRY")) {
      (*flow_xlow)[i] = xflux<upwinding>(a, f, i.xm());
      (*flow_zlow)[i] = zflux<upwinding>(a, f, i.zm());
    }
    BOUT_FOR(i, f.getRegion("RGN_NOBNDRY")) {
      result[i] =
          (*flow_xlow)[i] - (*flow_xlow)[i.xm()] + (*flow_zlow)[i] - (*flow_zlow)[i.zm()];
    }
  }
  result.setRegion("RGN_NOBNDRY");
  if constexpr (extra) {
    flow_xlow->setRegion("RGN_NOBNDRY");
    flow_zlow->setRegion("RGN_NOBNDRY");
    *flow_xlow /= volume;
    *flow_zlow /= volume;
  }
  return result / volume;
}

template <bool upwinding>
inline BoutReal Div_a_Grad_perp_FCI_structured::xflux(const Field3D& a, const Field3D& f,
                                                      const Ind3D& i) const {
  const auto ixp = i.xp();
  const auto dx = f[ixp] - f[i];
  BoutReal av = BoutNaN;
  if constexpr (upwinding) {
    av = dx > 0 ? a[ixp] : a[i];
  } else {
    av = 0.5 * (a[i] + a[ixp]);
  };
  const auto dz = 0.5 * (f[i.zp()] - f[i.zm()] + f[i.zp().xp()] - f[i.zm().xp()]);
  return -((fac_XX[i] * dx) + (fac_XZ[i] * dz)) * av;
}

template <bool upwinding>
inline BoutReal Div_a_Grad_perp_FCI_structured::zflux(const Field3D& a, const Field3D& f,
                                                      const Ind3D& i) const {
  const auto izp = i.zp();
  const auto dz = f[izp] - f[i];
  BoutReal av = BoutNaN;
  if constexpr (upwinding) {
    av = dz > 0 ? a[izp] : a[i];
  } else {
    av = 0.5 * (a[i] + a[izp]);
  }
  const auto dx = 0.5 * (f[i.xp()] - f[i.xm()] + f[izp.xp()] - f[izp.xm()]);
  return -((fac_ZX[i] * dx) + (fac_ZZ[i] * dz)) * av;
}

#if BOUT_HAS_PETSC
struct Div_a_Grad_perp_FCI_unstructured::Impl {
  /// Keeps the shared cell-space mapping alive
  std::shared_ptr<PetscOperators> ops;
  PetscCellToFacePointOperator grad;
  PetscCellToFacePointOperator value;
  PetscFacePointToCellOperator div;
  /// Optional (grids written by zoidberg.unstructured): the nodes on either
  /// side of each face point, for upwinding
  std::optional<PetscCellToFacePointOperator> left;
  std::optional<PetscCellToFacePointOperator> right;
  /// Optional: cell average of the flux vector -a grad f (R and Z components)
  std::optional<PetscFacePointToCellOperator> flowR;
  std::optional<PetscFacePointToCellOperator> flowZ;
  BoutReal rho_s0;

  /// Work vectors reused between calls (no PETSc allocations per call)
  struct Work {
    PetscCellVector a_cells, f_cells, cells_out;
    PetscFacePointVector grad_face, a_face, a_right, flux;
  };
  mutable std::optional<Work> work;

  Field3D apply(const Field3D& a, const Field3D& f, bool upwinding, Field3D* flow_R,
                Field3D* flow_Z) const;
};

namespace {
template <typename T>
Array<T> meshGetArray(Mesh* mesh, const std::string& name) {
  Array<T> result;
  if (mesh->get(result, name) != 0) {
    throw BoutException("Div_a_Grad_perp_FCI_unstructured requires array '{}'", name);
  }
  return result;
}

int meshGetInt(Mesh* mesh, const std::string& name) {
  if (not mesh->sourceHasVar(name)) {
    throw BoutException("Div_a_Grad_perp_FCI_unstructured requires integer '{}'", name);
  }
  int result{0};
  mesh->get(result, name);
  return result;
}

/// Pack the evolving cells of a field into an existing cell-space vector
void packCellsInto(const PetscCellMappingPtr& mapping, const Field3D& f,
                   PetscCellVector& out) {
  BOUT_DO_PETSC(VecSet(out.raw(), 0.0));
  PetscScalar* x{nullptr};
  BOUT_DO_PETSC(VecGetArray(out.raw(), &x));
  mapping->mapLocalField([&](PetscInt row, const Ind3D& i) { x[row] = f[i]; });
  BOUT_DO_PETSC(VecRestoreArray(out.raw(), &x));
}

/// Unpack a cell-space vector into a field without parallel slices
Field3D unpackCells(const PetscCellVector& vec, const Field3D& prototype) {
  const auto mapping = std::static_pointer_cast<const PetscCellMapping>(vec.getMapping());
  Field3D result{zeroFrom(prototype)};
  const PetscScalar* x{nullptr};
  BOUT_DO_PETSC(VecGetArrayRead(vec.raw(), &x));
  mapping->mapLocalField([&](PetscInt row, const Ind3D& i) { result[i] = x[row]; });
  BOUT_DO_PETSC(VecRestoreArrayRead(vec.raw(), &x));
  return result;
}
} // namespace
#else
struct Div_a_Grad_perp_FCI_unstructured::Impl {};
#endif

std::shared_ptr<Div_a_Grad_perp_FCI>
Div_a_Grad_perp_FCI_unstructured::create(Mesh* mesh, BoutReal rho_s0) {
  if (auto search = Div_a_Grad_perp_FCI_unstructured::op_cache.find(mesh);
      search != Div_a_Grad_perp_FCI_unstructured::op_cache.end()) {
    if (std::shared_ptr<Div_a_Grad_perp_FCI> spt = search->second.lock()) {
      return spt;
    }
    output_warn.write("\tPreformance Warning: DIV_A_GRAD_PERP needs to be re-created!\n");
  }
  auto dagp = std::make_shared<Div_a_Grad_perp_FCI_unstructured>(mesh, rho_s0);
  Div_a_Grad_perp_FCI_unstructured::op_cache[mesh] = dagp;
  return dagp;
}

Div_a_Grad_perp_FCI_unstructured::~Div_a_Grad_perp_FCI_unstructured() = default;

#if BOUT_HAS_PETSC
Div_a_Grad_perp_FCI_unstructured::Div_a_Grad_perp_FCI_unstructured(Mesh* mesh,
                                                                   BoutReal rho_s0) {
  auto ops = PetscOperators::Cache::get(mesh);
  const auto& cells = ops->cellMapping();

  // Each face point belongs to the rank that owns its owner cell. Older grids
  // (e.g. the TCV work-in-progress grids) have no ownership arrays: derive
  // them from the divergence matrix, owning each face point with the first
  // cell whose divergence uses it.
  int n_facepoints{0};
  Array<int> owner;
  if (mesh->sourceHasVar("dagp_us_facepoint_owner")) {
    n_facepoints = meshGetInt(mesh, "dagp_us_n_facepoints");
    owner = meshGetArray<int>(mesh, "dagp_us_facepoint_owner");
  } else {
    n_facepoints = meshGetInt(mesh, "dagp_us_div_ncols");
    const auto rows = meshGetArray<int>(mesh, "dagp_us_div_rows");
    const auto cols = meshGetArray<int>(mesh, "dagp_us_div_columns");
    owner = Array<int>(n_facepoints);
    std::fill(owner.begin(), owner.end(), -1);
    for (int cell = 0; cell + 1 < static_cast<int>(rows.size()); ++cell) {
      for (int index = rows[cell]; index < rows[cell + 1]; ++index) {
        if (owner[cols[index]] < 0) {
          owner[cols[index]] = cell;
        }
      }
    }
    // Face points without any divergence entry do not affect the result;
    // give them to the first cell so that every face point has an owner
    std::replace(owner.begin(), owner.end(), -1, 0);
  }
  if (owner.size() != n_facepoints) {
    throw BoutException("dagp_us_facepoint_owner has {} entries, expected {}",
                        owner.size(), n_facepoints);
  }
  std::vector<int> local_facepoints;
  for (int fp = 0; fp < n_facepoints; ++fp) {
    if (cells->ownsStored(owner[fp])) {
      local_facepoints.push_back(fp);
    }
  }
  const auto faces =
      std::make_shared<const PetscFacePointMapping>(n_facepoints, local_facepoints);

  // Columns of the cell-space matrices are stored cell numbers; grids may
  // omit the parallel-boundary virtual cells (numbered last), so ncols can
  // be smaller than the cell space. Face-point columns must match exactly.
  const auto read = [&](const std::string& name, int max_ncols, bool exact) {
    const int ncols = meshGetInt(mesh, name + "_ncols");
    if (ncols > max_ncols or (exact and ncols != max_ncols)) {
      throw BoutException("{}_ncols = {} does not match the grid ({})", name, ncols,
                          max_ncols);
    }
    return std::make_tuple(meshGetArray<int>(mesh, name + "_rows"),
                           meshGetArray<int>(mesh, name + "_columns"),
                           meshGetArray<BoutReal>(mesh, name + "_weights"));
  };

  const int n_cells = static_cast<int>(cells->globalSize());
  auto [grad_rows, grad_cols, grad_weights] = read("dagp_us_gradp", n_cells, false);
  auto [value_rows, value_cols, value_weights] = read("dagp_us_value", n_cells, false);
  auto [div_rows, div_cols, div_weights] = read("dagp_us_div", n_facepoints, true);

  // The grid data are in metres; convert to normalised units as the
  // structured operator does (its volume is divided by rho_s0^2)
  impl = std::make_unique<Impl>(Impl{
      ops, PetscCellToFacePointOperator(faces, cells, grad_rows, grad_cols, grad_weights),
      PetscCellToFacePointOperator(faces, cells, value_rows, value_cols, value_weights),
      PetscFacePointToCellOperator(cells, faces, div_rows, div_cols, div_weights)
          * SQ(rho_s0),
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, rho_s0});

  if (mesh->sourceHasVar("dagp_us_left_rows")) {
    auto [l_rows, l_cols, l_weights] = read("dagp_us_left", n_cells, false);
    auto [r_rows, r_cols, r_weights] = read("dagp_us_right", n_cells, false);
    impl->left.emplace(faces, cells, l_rows, l_cols, l_weights);
    impl->right.emplace(faces, cells, r_rows, r_cols, r_weights);
  }
  if (mesh->sourceHasVar("dagp_us_flowR_rows")) {
    auto [fr_rows, fr_cols, fr_weights] = read("dagp_us_flowR", n_facepoints, true);
    auto [fz_rows, fz_cols, fz_weights] = read("dagp_us_flowZ", n_facepoints, true);
    // Flux vector in normalised units: one length scale (rho_s0) fewer
    // than the divergence
    impl->flowR.emplace(
        PetscFacePointToCellOperator(cells, faces, fr_rows, fr_cols, fr_weights)
        * rho_s0);
    impl->flowZ.emplace(
        PetscFacePointToCellOperator(cells, faces, fz_rows, fz_cols, fz_weights)
        * rho_s0);
  }
}

Field3D Div_a_Grad_perp_FCI_unstructured::Impl::apply(const Field3D& a, const Field3D& f,
                                                      bool upwinding, Field3D* flow_R,
                                                      Field3D* flow_Z) const {
  ASSERT1_FIELDS_COMPATIBLE(a, f);
  const auto& cells = ops->cellMapping();
  if (not work) {
    const auto faces =
        std::static_pointer_cast<const PetscFacePointMapping>(grad.outMapping());
    work.emplace(Work{PetscCellVector(cells), PetscCellVector(cells),
                      PetscCellVector(cells), PetscFacePointVector(faces),
                      PetscFacePointVector(faces), PetscFacePointVector(faces),
                      PetscFacePointVector(faces)});
  }
  auto& w = *work;
  packCellsInto(cells, a, w.a_cells);
  packCellsInto(cells, f, w.f_cells);
  const auto& a_cells = w.a_cells;
  const auto& grad_face = w.grad_face;
  grad.apply(w.f_cells, w.grad_face);

  auto& a_face = w.a_face;
  if (upwinding) {
    if (not(left and right)) {
      throw BoutException("Upwinding on unstructured meshes needs dagp_us_left/right "
                          "in the grid file (regenerate with zoidberg.unstructured)");
    }
    // As in the structured operator, take a from the side with the larger f
    // (where the diffusive flux comes from). The face gradient S.grad f is
    // positive if f increases towards the right node. The value is the
    // reconstruction from the upwind side stored in the grid file
    // (dagp_us_left/right).
    left->apply(a_cells, a_face);
    right->apply(a_cells, w.a_right);
    const auto& a_right = w.a_right;
    PetscScalar* av{nullptr};
    const PetscScalar* ar{nullptr};
    const PetscScalar* gf{nullptr};
    BOUT_DO_PETSC(VecGetArray(a_face.raw(), &av));
    BOUT_DO_PETSC(VecGetArrayRead(a_right.raw(), &ar));
    BOUT_DO_PETSC(VecGetArrayRead(grad_face.raw(), &gf));
    PetscInt n_local{0};
    BOUT_DO_PETSC(VecGetLocalSize(a_face.raw(), &n_local));
    for (PetscInt i = 0; i < n_local; ++i) {
      if (gf[i] > 0) {
        av[i] = ar[i];
      }
    }
    BOUT_DO_PETSC(VecRestoreArray(a_face.raw(), &av));
    BOUT_DO_PETSC(VecRestoreArrayRead(a_right.raw(), &ar));
    BOUT_DO_PETSC(VecRestoreArrayRead(grad_face.raw(), &gf));
  } else {
    value.apply(a_cells, a_face);
  }
  auto& flux = w.flux;
  BOUT_DO_PETSC(VecPointwiseMult(flux.raw(), a_face.raw(), grad_face.raw()));

  if (flow_R != nullptr) {
    if (flowR and flowZ) {
      flowR->apply(flux, w.cells_out);
      *flow_R = unpackCells(w.cells_out, f);
      flowZ->apply(flux, w.cells_out);
      *flow_Z = unpackCells(w.cells_out, f);
    } else {
      *flow_R = zeroFrom(f);
      *flow_Z = zeroFrom(f);
    }
    flow_R->setRegion("RGN_NOBNDRY");
    flow_Z->setRegion("RGN_NOBNDRY");
  }
  div.apply(flux, w.cells_out);
  auto result = unpackCells(w.cells_out, f);
  result.setRegion("RGN_NOBNDRY");
  return result;
}

Field3D Div_a_Grad_perp_FCI_unstructured::operator()(const Field3D& a, const Field3D& f,
                                                     bool upwinding) const {
  return impl->apply(a, f, upwinding, nullptr, nullptr);
}

Field3D Div_a_Grad_perp_FCI_unstructured::operator()(const Field3D& a, const Field3D& f,
                                                     Field3D& flow_xlow,
                                                     Field3D& flow_zlow,
                                                     bool upwinding) const {
  return impl->apply(a, f, upwinding, &flow_xlow, &flow_zlow);
}
#else
Div_a_Grad_perp_FCI_unstructured::Div_a_Grad_perp_FCI_unstructured(
    [[maybe_unused]] Mesh* mesh, [[maybe_unused]] BoutReal rho_s0) {
  throw BoutException("Compile BOUT++ with PETSc to run on unstructured meshes!");
}

Field3D
Div_a_Grad_perp_FCI_unstructured::operator()([[maybe_unused]] const Field3D& a,
                                             [[maybe_unused]] const Field3D& f,
                                             [[maybe_unused]] bool upwinding) const {
  throw BoutException("Compile BOUT++ with PETSc to run on unstructured meshes!");
}

Field3D Div_a_Grad_perp_FCI_unstructured::operator()(
    [[maybe_unused]] const Field3D& a, [[maybe_unused]] const Field3D& f,
    [[maybe_unused]] Field3D& flow_xlow, [[maybe_unused]] Field3D& flow_zlow,
    [[maybe_unused]] bool upwinding) const {
  throw BoutException("Compile BOUT++ with PETSc to run on unstructured meshes!");
}
#endif

} // Namespace FV
