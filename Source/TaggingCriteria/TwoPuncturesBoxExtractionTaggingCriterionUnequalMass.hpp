/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef TWOPUNCTURESBOXEXTRACTIONTAGGINGCRITERIONUNEQUALMASS_HPP_
#define TWOPUNCTURESBOXEXTRACTIONTAGGINGCRITERIONUNEQUALMASS_HPP_

#include "Cell.hpp"
#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "SphericalExtraction.hpp"
#include "Tensor.hpp"
#include <algorithm>

class TwoPuncturesBoxExtractionTaggingCriterionUnequalMass
{
  protected:
    const double m_dx;
    const int m_level;
    const spherical_extraction_params_t m_params;
    const bool m_activate_extraction;
    const std::vector<double> m_puncture_masses;
    const std::vector<std::array<double, CH_SPACEDIM>> &m_puncture_coords;
    const double m_grid_radius; // half-width of the finest box around each
                                // puncture in units of its mass
    const std::array<int, 2> m_punctures_max_levels; // finest level around
                                                     // each puncture
    const double m_puncture_min_separation; // the minimum coordinate separation
                                            // to tag around the punctures
                                            // separately

  public:
    TwoPuncturesBoxExtractionTaggingCriterionUnequalMass(
        const double dx, const int a_level,
        const spherical_extraction_params_t a_params,
        const std::vector<std::array<double, CH_SPACEDIM>> &a_puncture_coords,
        const bool activate_extraction,
        const std::vector<double> a_puncture_masses,
        const double a_grid_radius,
        const std::array<int, 2> a_punctures_max_levels,
        const double a_puncture_min_separation = 1e-3)
        : m_dx(dx), m_level(a_level), m_params(a_params),
          m_activate_extraction(activate_extraction),
          m_puncture_masses(a_puncture_masses),
          m_puncture_coords(a_puncture_coords), m_grid_radius(a_grid_radius),
          m_punctures_max_levels(a_punctures_max_levels),
          m_puncture_min_separation(a_puncture_min_separation)
    {
        // check that the number of punctures is consistent
        CH_assert(m_puncture_coords.size() == 2);
        CH_assert(m_puncture_masses.size() == 2);
    };

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        data_t criterion = 0.0;

        double puncture_separation_squared = 0.0;
        FOR(i)
        {
            double displacement =
                m_puncture_coords[0][i] - m_puncture_coords[1][i];
            puncture_separation_squared += displacement * displacement;
        }
        const double puncture_separation = sqrt(puncture_separation_squared);
        const double sum_masses = m_puncture_masses[0] + m_puncture_masses[1];

        // punctures still far enough apart so tag around each one separately
        if (puncture_separation > m_puncture_min_separation)
        {
            for (int ipuncture = 0; ipuncture < 2; ++ipuncture)
            {
                // no finer levels around this puncture
                const int puncture_max_level = m_punctures_max_levels[ipuncture];
                if (m_level >= puncture_max_level)
                    continue;

                // we want each level to be double the size of the one inside
                // it, counting from this puncture's own max level, so the
                // finest box has half-width m_grid_radius * mass
                const double factor =
                    pow(2.0, puncture_max_level - m_level - 1);

                // where am i?
                const Coordinates<data_t> coords(current_cell, m_dx,
                                                 m_puncture_coords[ipuncture]);
                const data_t max_abs_xy =
                    simd_max(abs(coords.x), abs(coords.y));
                const data_t max_abs_xyz = simd_max(max_abs_xy, abs(coords.z));
                auto regrid = simd_compare_lt(
                    max_abs_xyz,
                    m_grid_radius * factor * m_puncture_masses[ipuncture]);
                criterion = simd_conditional(regrid, 100.0, criterion);
            }
        }

        // if punctures are close enough together tag cells around the centre
        // of mass for the merger BH, up to the coarser of the two puncture max
        // levels
        const int merger_max_level =
            std::min(m_punctures_max_levels[0], m_punctures_max_levels[1]);
        if (puncture_separation <= sum_masses && m_level < merger_max_level)
        {
            const double factor = pow(2.0, merger_max_level - m_level - 1);

            // where am i?
            std::array<double, CH_SPACEDIM> center_of_mass;
            FOR(i)
            {
                center_of_mass[i] =
                    (m_puncture_masses[0] * m_puncture_coords[0][i] +
                     m_puncture_masses[1] * m_puncture_coords[1][i]) /
                    sum_masses;
            }
            const Coordinates<data_t> coords(current_cell, m_dx,
                                             center_of_mass);
            const data_t max_abs_xy = simd_max(abs(coords.x), abs(coords.y));
            const data_t max_abs_xyz = simd_max(max_abs_xy, abs(coords.z));
            auto regrid =
                simd_compare_lt(max_abs_xyz, 1.5 * factor * sum_masses);
            criterion = simd_conditional(regrid, 100.0, criterion);
        }

        // if extracting weyl data at a given radius, enforce a given resolution
        // there
        if (m_activate_extraction)
        {
            for (int iradius = 0; iradius < m_params.num_extraction_radii;
                 ++iradius)
            {
                // regrid if within extraction level and not at required
                // refinement
                if (m_level < m_params.extraction_levels[iradius])
                {
                    const Coordinates<data_t> coords(current_cell, m_dx,
                                                     m_params.center);
                    const data_t r = coords.get_radius();
                    // add a 20% buffer to extraction zone so not too near to
                    // boundary
                    auto regrid = simd_compare_lt(
                        r, 1.2 * m_params.extraction_radii[iradius]);
                    criterion = simd_conditional(regrid, 100.0, criterion);
                }
            }
        }
        // Write back into the flattened Chombo box
        current_cell.store_vars(criterion, 0);
    }
};

#endif /* TWOPUNCTURESBOXEXTRACTIONTAGGINGCRITERIONUNEQUALMASS_HPP_ */
