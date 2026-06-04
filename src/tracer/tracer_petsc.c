#ifndef TRACER_PETSC_H
#define TRACER_PETSC_H

#include <petscsys.h>
#include <private/rdymathimpl.h>
#include <private/rdysweimpl.h>

#include "tracer_roe_flux_petsc.h"

typedef struct {
  const RDyPhysicsSD *sediment;
  PetscInt            num_bed_layers;
  PetscReal           active_layer_thickness;
  PetscReal          *bed_init_thickness;
  PetscReal          *bed_mass;
  PetscReal          *active_layer_concentration;
  PetscReal          *bed_porosity;
  PetscReal          *bed_rho_s;
} TracerBedModel;

#define BED_INDEX(layer, cell, s, num_cells, num_tracers_comp) (((layer) * (num_cells) + (cell)) * (num_tracers_comp) + (s))

static PetscErrorCode CheckTracerSedimentConfig(const RDyPhysicsSD *sediment, PetscInt num_tracers_comp, MPI_Comm comm) {
  PetscFunctionBeginUser;

  PetscCheck(sediment->num_classes == num_tracers_comp, comm, PETSC_ERR_USER,
             "Sediment configuration has %" PetscInt_FMT " classes, but tracer operator has %" PetscInt_FMT, sediment->num_classes,
             num_tracers_comp);
  PetscCheck(sediment->bed.substrate_layers_count > 0, comm, PETSC_ERR_USER, "Sediment bed must have at least one substrate layer");

  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscInt TracerLayerPropertyIndex(PetscInt layer) { return (layer == 0) ? 0 : (layer - 1); }

static inline PetscReal TracerLayerConcentration(const TracerBedModel *bed, PetscInt layer) {
  return bed->sediment->bed.substrate_layers[TracerLayerPropertyIndex(layer)].concentration;
}

static PetscReal ClampTracerActiveLayerConcentration(const TracerBedModel *bed, PetscReal active_conc) {
  const PetscInt nconc = bed->sediment->bed.substrate_layers_count;
  const RDySedimentSubstrateLayer *layers = bed->sediment->bed.substrate_layers;
  if (active_conc < layers[0].concentration) active_conc = layers[0].concentration;
  if (active_conc > layers[nconc - 1].concentration) active_conc = layers[nconc - 1].concentration;
  return active_conc;
}

static PetscReal ComputeTracerLayerTotalMass(const TracerBedModel *bed, PetscInt cell, PetscInt layer, PetscInt num_cells,
                                             PetscInt num_tracers_comp) {
  PetscReal M_tot = 0.0;
  for (PetscInt s = 0; s < num_tracers_comp; ++s) {
    PetscReal m = bed->bed_mass[BED_INDEX(layer, cell, s, num_cells, num_tracers_comp)];
    if (!PetscIsInfOrNanReal(m) && m > 0.0) M_tot += m;
  }
  return M_tot;
}

static PetscReal ComputeTracerActiveLayerConcentration(const TracerBedModel *bed, PetscInt cell, PetscInt num_cells,
                                                       PetscInt num_tracers_comp) {
  (void)num_cells;
  (void)num_tracers_comp;
  return ClampTracerActiveLayerConcentration(bed, bed->active_layer_concentration[cell]);
}

static void InterpolateTracerActiveLayerErosionProperties(const TracerBedModel *bed, PetscReal active_conc, PetscReal *tau_e_active,
                                                          PetscReal *kp_active) {
  const PetscInt nconc = bed->sediment->bed.substrate_layers_count;
  const RDySedimentSubstrateLayer *layers = bed->sediment->bed.substrate_layers;

  if (active_conc <= layers[0].concentration) {
    *tau_e_active = layers[0].critical_erosion_shear_stress;
    *kp_active    = layers[0].partheniades_constant;
    return;
  }
  if (active_conc >= layers[nconc - 1].concentration) {
    *tau_e_active = layers[nconc - 1].critical_erosion_shear_stress;
    *kp_active    = layers[nconc - 1].partheniades_constant;
    return;
  }

  for (PetscInt k = 1; k < nconc; ++k) {
    PetscReal c0 = layers[k - 1].concentration;
    PetscReal c1 = layers[k].concentration;
    if (active_conc <= c1) {
      PetscReal alpha = (active_conc - c0) / (c1 - c0);
      *tau_e_active   = layers[k - 1].critical_erosion_shear_stress +
                      alpha * (layers[k].critical_erosion_shear_stress - layers[k - 1].critical_erosion_shear_stress);
      *kp_active = layers[k - 1].partheniades_constant +
                   alpha * (layers[k].partheniades_constant - layers[k - 1].partheniades_constant);
      return;
    }
  }

  *tau_e_active = layers[nconc - 1].critical_erosion_shear_stress;
  *kp_active    = layers[nconc - 1].partheniades_constant;
}

static void GetTracerLayerErosionProperties(const TracerBedModel *bed, PetscInt layer, PetscReal active_conc, PetscReal *tau_e_layer,
                                            PetscReal *kp_layer) {
  if (layer == 0) {
    InterpolateTracerActiveLayerErosionProperties(bed, active_conc, tau_e_layer, kp_layer);
  } else {
    PetscInt idx = TracerLayerPropertyIndex(layer);
    *tau_e_layer = bed->sediment->bed.substrate_layers[idx].critical_erosion_shear_stress;
    *kp_layer    = bed->sediment->bed.substrate_layers[idx].partheniades_constant;
  }
}

static void ComputeTracerErodedMassByClass(TracerBedModel *bed, PetscInt cell, PetscInt num_cells, PetscInt num_tracers_comp, PetscReal h,
                                           PetscReal h_ero_min, PetscReal tau_b, PetscReal dt, PetscReal *eroded_mass_by_class) {
  for (PetscInt s = 0; s < num_tracers_comp; ++s) eroded_mass_by_class[s] = 0.0;

  if (h < h_ero_min || dt <= 0.0) return;

  PetscReal remaining_dt = dt;
  PetscReal active_conc  = ComputeTracerActiveLayerConcentration(bed, cell, num_cells, num_tracers_comp);

  for (PetscInt layer = 0; layer < bed->num_bed_layers && remaining_dt > 0.0; ++layer) {
    PetscReal Mtot_layer = ComputeTracerLayerTotalMass(bed, cell, layer, num_cells, num_tracers_comp);
    if (Mtot_layer <= 0.0) continue;

    PetscReal tau_e_layer = 0.0;
    PetscReal kp_layer    = 0.0;
    GetTracerLayerErosionProperties(bed, layer, active_conc, &tau_e_layer, &kp_layer);

    if (tau_e_layer <= 0.0 || tau_b <= tau_e_layer) break;

    PetscReal erosion_flux = kp_layer * (tau_b - tau_e_layer) / tau_e_layer;
    if (erosion_flux <= 0.0) break;

    PetscReal erodible_mass = PetscMin(Mtot_layer, erosion_flux * remaining_dt);
    if (erodible_mass <= 0.0) break;

    for (PetscInt s = 0; s < num_tracers_comp; ++s) {
      PetscInt  idx_layer = BED_INDEX(layer, cell, s, num_cells, num_tracers_comp);
      PetscReal M_layer_s = bed->bed_mass[idx_layer];
      if (PetscIsInfOrNanReal(M_layer_s) || M_layer_s < 0.0) M_layer_s = 0.0;

      PetscReal frac_s = M_layer_s / Mtot_layer;
      PetscReal dm_s   = erodible_mass * frac_s;
      if (dm_s > bed->bed_mass[idx_layer]) dm_s = bed->bed_mass[idx_layer];

      bed->bed_mass[idx_layer] -= dm_s;
      eroded_mass_by_class[s] += dm_s;
    }

    PetscReal time_used = erodible_mass / erosion_flux;
    remaining_dt -= time_used;
    if (erodible_mass < Mtot_layer) break;
  }
}

static PetscErrorCode InitializeTracerBedModel(RDyMesh *mesh, const RDyPhysicsSD *sediment, PetscInt num_tracers_comp, MPI_Comm comm,
                                               TracerBedModel *bed) {
  PetscFunctionBeginUser;

  PetscCall(CheckTracerSedimentConfig(sediment, num_tracers_comp, comm));

  PetscInt ncells = mesh->num_cells;

  bed->sediment               = sediment;
  bed->active_layer_thickness = sediment->bed.active_layer_thickness;
  bed->num_bed_layers         = sediment->bed.substrate_layers_count + 1;

  PetscCall(PetscCalloc1(bed->num_bed_layers, &bed->bed_porosity));
  PetscCall(PetscCalloc1(bed->num_bed_layers, &bed->bed_init_thickness));
  PetscCall(PetscCalloc1(num_tracers_comp, &bed->bed_rho_s));
  PetscCall(PetscCalloc1(ncells, &bed->active_layer_concentration));

  bed->bed_init_thickness[0] = bed->active_layer_thickness;
  for (PetscInt layer = 1; layer < bed->num_bed_layers; ++layer) {
    bed->bed_init_thickness[layer] = sediment->bed.substrate_layers[layer - 1].initial_thickness;
  }
  for (PetscInt s = 0; s < num_tracers_comp; ++s) {
    bed->bed_rho_s[s] = sediment->classes[s].density;
  }

  PetscReal rho_bulk = 0.0;
  for (PetscInt s = 0; s < num_tracers_comp; ++s) {
    rho_bulk += sediment->classes[s].initial_fraction * bed->bed_rho_s[s];
  }
  PetscCheck(rho_bulk > 0.0, comm, PETSC_ERR_USER, "Bulk sediment density must be positive");

  for (PetscInt layer = 0; layer < bed->num_bed_layers; ++layer) {
    PetscReal concentration = TracerLayerConcentration(bed, layer);
    PetscCheck(concentration > 0.0, comm, PETSC_ERR_USER,
               "Sediment layer concentration for bed layer %" PetscInt_FMT " must be > 0", layer);
    PetscCheck(concentration <= rho_bulk, comm, PETSC_ERR_USER,
               "Sediment layer concentration for bed layer %" PetscInt_FMT "=%g exceeds bulk grain density %g", layer, (double)concentration,
               (double)rho_bulk);
    bed->bed_porosity[layer] = 1.0 - concentration / rho_bulk;
  }

  PetscInt nbed = bed->num_bed_layers * ncells * num_tracers_comp;
  PetscCall(PetscCalloc1(nbed, &bed->bed_mass));
  for (PetscInt c = 0; c < ncells; ++c) {
    bed->active_layer_concentration[c] = sediment->bed.substrate_layers[0].concentration;
    for (PetscInt layer = 0; layer < bed->num_bed_layers; ++layer) {
      PetscReal hL           = bed->bed_init_thickness[layer];
      PetscReal layer_mass   = TracerLayerConcentration(bed, layer) * hL;
      for (PetscInt s = 0; s < num_tracers_comp; ++s) {
        PetscReal mL  = layer_mass * sediment->classes[s].initial_fraction;
        bed->bed_mass[BED_INDEX(layer, c, s, ncells, num_tracers_comp)] = mL;
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode UpdateTracerBedActiveLayer(RDyMesh *mesh, PetscInt num_tracers_comp, TracerBedModel *bed) {
  PetscFunctionBeginUser;

  PetscReal *h_layer = NULL;
  PetscCall(PetscMalloc1(bed->num_bed_layers, &h_layer));

  for (PetscInt c = 0; c < mesh->num_cells; ++c) {
    PetscReal active_conc = ComputeTracerActiveLayerConcentration(bed, c, mesh->num_cells, num_tracers_comp);
    PetscReal active_mass = ComputeTracerLayerTotalMass(bed, c, 0, mesh->num_cells, num_tracers_comp);
    if (bed->num_bed_layers > 1) active_conc = PetscMax(active_conc, TracerLayerConcentration(bed, 1));

    for (PetscInt layer = 0; layer < bed->num_bed_layers; ++layer) {
      PetscReal M_tot    = 0.0;
      PetscReal rho_bulk = 0.0;
      for (PetscInt s = 0; s < num_tracers_comp; ++s) {
        PetscReal m = bed->bed_mass[BED_INDEX(layer, c, s, mesh->num_cells, num_tracers_comp)];
        if (m < 0.0) m = 0.0;
        M_tot += m;
      }
      if (M_tot <= 0.0) {
        h_layer[layer] = 0.0;
        continue;
      }
      for (PetscInt s = 0; s < num_tracers_comp; ++s) {
        PetscReal m = bed->bed_mass[BED_INDEX(layer, c, s, mesh->num_cells, num_tracers_comp)];
        if (m < 0.0) m = 0.0;
        rho_bulk += (m / M_tot) * bed->bed_rho_s[s];
      }
      h_layer[layer] = M_tot / (rho_bulk * (1.0 - bed->bed_porosity[layer]));
    }

    PetscReal dht = h_layer[0] - bed->active_layer_thickness;
    if (dht > 0.0 && h_layer[0] > 0.0 && bed->num_bed_layers > 1) {
      PetscReal frac = dht / h_layer[0];
      for (PetscInt s = 0; s < num_tracers_comp; ++s) {
        PetscInt idx_active = BED_INDEX(0, c, s, mesh->num_cells, num_tracers_comp);
        PetscInt idx_next   = BED_INDEX(1, c, s, mesh->num_cells, num_tracers_comp);
        PetscReal dm        = frac * bed->bed_mass[idx_active];
        if (dm > bed->bed_mass[idx_active]) dm = bed->bed_mass[idx_active];
        bed->bed_mass[idx_active] -= dm;
        bed->bed_mass[idx_next] += dm;
      }
    } else if (dht < 0.0) {
      PetscReal need = -dht;
      for (PetscInt layer = 1; layer < bed->num_bed_layers && need > 0.0; ++layer) {
        PetscReal ratio_xkv = (1.0 - bed->bed_porosity[layer]) / (1.0 - bed->bed_porosity[0]);
        PetscReal eff_h_L   = h_layer[layer] * ratio_xkv;
        if (eff_h_L <= 0.0) continue;
        PetscReal donor_conc = TracerLayerConcentration(bed, layer);
        if (need >= eff_h_L) {
          PetscReal dm_tot = 0.0;
          for (PetscInt s = 0; s < num_tracers_comp; ++s) {
            PetscInt idx_active = BED_INDEX(0, c, s, mesh->num_cells, num_tracers_comp);
            PetscInt idx_layer  = BED_INDEX(layer, c, s, mesh->num_cells, num_tracers_comp);
            PetscReal dm        = bed->bed_mass[idx_layer];
            bed->bed_mass[idx_layer] -= dm;
            bed->bed_mass[idx_active] += dm;
            dm_tot += dm;
          }
          if (dm_tot > 0.0) {
            if (active_mass + dm_tot > 0.0) {
              active_conc = (active_conc * active_mass + donor_conc * dm_tot) / (active_mass + dm_tot);
            } else {
              active_conc = donor_conc;
            }
            active_mass += dm_tot;
          }
          need -= eff_h_L;
        } else {
          PetscReal frac = need / eff_h_L;
          PetscReal dm_tot = 0.0;
          for (PetscInt s = 0; s < num_tracers_comp; ++s) {
            PetscInt idx_active = BED_INDEX(0, c, s, mesh->num_cells, num_tracers_comp);
            PetscInt idx_layer  = BED_INDEX(layer, c, s, mesh->num_cells, num_tracers_comp);
            PetscReal dm        = frac * bed->bed_mass[idx_layer];
            if (dm > bed->bed_mass[idx_layer]) dm = bed->bed_mass[idx_layer];
            bed->bed_mass[idx_layer] -= dm;
            bed->bed_mass[idx_active] += dm;
            dm_tot += dm;
          }
          if (dm_tot > 0.0) {
            if (active_mass + dm_tot > 0.0) {
              active_conc = (active_conc * active_mass + donor_conc * dm_tot) / (active_mass + dm_tot);
            } else {
              active_conc = donor_conc;
            }
            active_mass += dm_tot;
          }
          need = 0.0;
        }
      }
    }

    bed->active_layer_concentration[c] = ClampTracerActiveLayerConcentration(bed, active_conc);
  }

  PetscCall(PetscFree(h_layer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyTracerBedModel(TracerBedModel *bed) {
  PetscFunctionBegin;
  PetscCall(PetscFree(bed->bed_init_thickness));
  PetscCall(PetscFree(bed->bed_mass));
  PetscCall(PetscFree(bed->active_layer_concentration));
  PetscCall(PetscFree(bed->bed_porosity));
  PetscCall(PetscFree(bed->bed_rho_s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Allocates memory for prognostic (h/hu/hv/hci) and diagnostic (u/v/ci) variables stored at
///        cell centers for tracers dynamics
/// @param [in]  num_states        number of states
/// @param [in]  num_flow_comp     number of components for the flow equation
/// @param [in]  num_tracers_comp number of components for the tracers dynamics equation
/// @param [out] *data             a TracerRiemannStateData
/// @return                        0 on success, or a non-zero error code on failure
static PetscErrorCode CreateTracerRiemannStateData(const PetscInt num_states, const PetscInt num_flow_comp, const PetscInt num_tracers_comp,
                                                   TracerRiemannStateData *data) {
  PetscFunctionBegin;

  data->num_states       = num_states;
  data->num_flow_comp    = num_flow_comp;
  data->num_tracers_comp = num_tracers_comp;

  PetscCall(PetscCalloc1(num_states, &data->h));
  PetscCall(PetscCalloc1(num_states, &data->hu));
  PetscCall(PetscCalloc1(num_states, &data->hv));
  PetscCall(PetscCalloc1(num_states, &data->u));
  PetscCall(PetscCalloc1(num_states, &data->v));

  PetscCall(PetscCalloc1(num_states * num_tracers_comp, &data->hci));
  PetscCall(PetscCalloc1(num_states * num_tracers_comp, &data->ci));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Deallocates memory for a struct that stores prognostic and diagnostic variables
///        at cell centers
/// @param [inout] data a TracerRiemannStateData that deallocated
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode DestroyTracerRiemannStateData(TracerRiemannStateData data) {
  PetscFunctionBegin;

  data.num_states       = 0;
  data.num_flow_comp    = 0;
  data.num_tracers_comp = 0;
  PetscCall(PetscFree(data.h));
  PetscCall(PetscFree(data.hu));
  PetscCall(PetscFree(data.hv));
  PetscCall(PetscFree(data.hci));
  PetscCall(PetscFree(data.u));
  PetscCall(PetscFree(data.v));
  PetscCall(PetscFree(data.ci));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Deallocates memory for a struct that stores diagnostic variables and geometric mesh
///        attributes at cell edges
/// @param [inout] data a TracerRiemannEdgeData that deallocated
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode DestroyTracerRiemannEdgeData(TracerRiemannEdgeData data) {
  PetscFunctionBegin;

  data.num_edges        = 0;
  data.num_flow_comp    = 0;
  data.num_tracers_comp = 0;

  PetscCall(PetscFree(data.cn));
  PetscCall(PetscFree(data.sn));
  PetscCall(PetscFree(data.fluxes));
  PetscCall(PetscFree(data.amax));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Allocates memory for diagnostic variables and geometric mesh attributes at cell edges
///        cell centers for tracers dynamics
/// @param [in]  num_edges         number of edges
/// @param [in]  num_flow_comp     number of components for the flow equation
/// @param [in]  num_tracers_comp number of components for the tracers dynamics equation
/// @param [out] *data             a TracerRiemannEdgeData
/// @return                        0 on success, or a non-zero error code on failure
static PetscErrorCode CreateTracerRiemannEdgeData(PetscInt num_edges, PetscInt num_flow_comp, PetscInt num_tracers_comp,
                                                  TracerRiemannEdgeData *data) {
  PetscFunctionBegin;

  data->num_edges        = num_edges;
  data->num_flow_comp    = num_flow_comp;
  data->num_tracers_comp = num_tracers_comp;

  PetscCall(PetscCalloc1(num_edges, &data->cn));
  PetscCall(PetscCalloc1(num_edges, &data->sn));

  PetscCall(PetscCalloc1(num_edges * (num_flow_comp + num_tracers_comp), &data->fluxes));
  PetscCall(PetscCalloc1(num_edges, &data->amax));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Computes diagnostic variables (u/v/ci) from prognostic variables (h/hu/hv/hci)
/// @param [in]  tiny_h  a height threshold for determining wet/dry cell
/// @param [out] *data   a TracerRiemannStateData
/// @return              0 on success, or a non-zero error code on failure
static PetscErrorCode ComputeRiemannVelocitiesAndConcentration(const PetscReal tiny_h, const PetscReal h_anuga,
                                                               TracerRiemannStateData *data) {
  PetscFunctionBeginUser;

  PetscReal denom;
  PetscInt index;
  for (PetscInt n = 0; n < data->num_states; n++) {
    if (data->h[n] < tiny_h) {
      data->u[n] = 0.0;
      data->v[n] = 0.0;
      for (PetscInt s = 0; s < data->num_tracers_comp; s++) {
        index           = n * data->num_tracers_comp + s;
        data->ci[index] = 0.0;
      }
    } else {
      denom      = Square(data->h[n]) + Square(h_anuga);
      data->u[n] = data->hu[n] * data->h[n] / denom;
      data->v[n] = data->hv[n] * data->h[n] / denom;
      for (PetscInt s = 0; s < data->num_tracers_comp; s++) {
        index           = n * data->num_tracers_comp + s;
        data->ci[index] = data->hci[index] / data->h[n];
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------
// Interior Flux Operator
//------------------------

typedef struct {
  RDyNumericsRiemann     riemann;       // riemann solver type
  RDyMesh               *mesh;          // domain mesh
  PetscReal              tiny_h;        // minimum water height for wet conditions
  PetscReal              h_anuga_regular;
  TracerRiemannStateData left_states;   // "left" riemann states on interior edges
  TracerRiemannStateData right_states;  // "right" riemann states on interior edges
  TracerRiemannEdgeData  edges;         // riemann fluxes on interior edges
  OperatorDiagnostics   *diagnostics;   // courant number, etc
} TracerInteriorFluxOperator;

/// @brief Computes the fluxes through the interior edges of mesh locally owned and
///        adds contribution in f_global Vec.
/// @param [in] context  a TracerInteriorFluxOperator
/// @param [in] fields   a PetscOperatorFields ! FIXME: Can possibly be deleted
/// @param [in] dt       time step             ! FIXME: Can possibly be deleted
/// @param [in] u_local  a Vec containing values for locally-owned and ghost cells
/// @param [in] f_global a Vec for storing RHS contrinution from interior edges
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode ApplyTracerInteriorFlux(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  TracerInteriorFluxOperator *interior_flux_op = context;

  RDyMesh  *mesh  = interior_flux_op->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  // get pointers to vector data
  PetscScalar *u_ptr, *f_ptr;
  PetscCall(VecGetArray(u_local, &u_ptr));
  PetscCall(VecGetArray(f_global, &f_ptr));

  TracerRiemannStateData *datal        = &interior_flux_op->left_states;
  TracerRiemannStateData *datar        = &interior_flux_op->right_states;
  TracerRiemannEdgeData  *data_edge    = &interior_flux_op->edges;
  PetscReal              *sn_vec_int   = data_edge->sn;
  PetscReal              *cn_vec_int   = data_edge->cn;
  PetscReal              *amax_vec_int = data_edge->amax;
  PetscReal              *flux_vec_int = data_edge->fluxes;

  PetscInt num_flow_comp    = datal->num_flow_comp;
  PetscInt num_tracers_comp = datal->num_tracers_comp;

  PetscInt n_dof;
  PetscCall(VecGetBlockSize(u_local, &n_dof));
  PetscCheck(n_dof == num_flow_comp + num_tracers_comp, comm, PETSC_ERR_USER,
             "Mismatch in number of dof in local vector (%" PetscInt_FMT ") and flow + tracers (%" PetscInt_FMT ")", n_dof,
             num_flow_comp + num_tracers_comp);

  // Collect the h/hu/hv/hci for left and right cells to compute u/v/ci
  for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
    PetscInt edge_id             = edges->internal_edge_ids[e];
    PetscInt left_local_cell_id  = edges->cell_ids[2 * edge_id];
    PetscInt right_local_cell_id = edges->cell_ids[2 * edge_id + 1];

    if (right_local_cell_id != -1) {
      datal->h[e]  = u_ptr[n_dof * left_local_cell_id + 0];
      datal->hu[e] = u_ptr[n_dof * left_local_cell_id + 1];
      datal->hv[e] = u_ptr[n_dof * left_local_cell_id + 2];

      datar->h[e]  = u_ptr[n_dof * right_local_cell_id + 0];
      datar->hu[e] = u_ptr[n_dof * right_local_cell_id + 1];
      datar->hv[e] = u_ptr[n_dof * right_local_cell_id + 2];

      for (PetscInt s = 0; s < num_tracers_comp; s++) {
        datal->hci[e * num_tracers_comp + s] = u_ptr[n_dof * left_local_cell_id + 3 + s];
        datar->hci[e * num_tracers_comp + s] = u_ptr[n_dof * right_local_cell_id + 3 + s];
      }
    }
  }

  // compute diagnostic quantities
  const PetscReal tiny_h  = interior_flux_op->tiny_h;
  const PetscReal h_anuga = interior_flux_op->h_anuga_regular;
  PetscCall(ComputeRiemannVelocitiesAndConcentration(tiny_h, h_anuga, datal));
  PetscCall(ComputeRiemannVelocitiesAndConcentration(tiny_h, h_anuga, datar));

  // call Riemann solver
  switch (interior_flux_op->riemann) {
    case RIEMANN_ROE:
      PetscCall(ComputeTracerRoeFlux(datal, datar, sn_vec_int, cn_vec_int, flux_vec_int, amax_vec_int));
      break;
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Unsupported Riemann solver");
  }

  // accummulate the flux values in the global flux vector
  for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
    PetscInt edge_id             = edges->internal_edge_ids[e];
    PetscInt left_local_cell_id  = edges->cell_ids[2 * edge_id];
    PetscInt right_local_cell_id = edges->cell_ids[2 * edge_id + 1];

    if (right_local_cell_id != -1) {  // internal edge
      PetscReal edge_len = edges->lengths[edge_id];

      PetscReal hl = u_ptr[n_dof * left_local_cell_id + 0];
      PetscReal hr = u_ptr[n_dof * right_local_cell_id + 0];

      if (!(hr < tiny_h && hl < tiny_h)) {  // either cell is "wet"
        PetscReal areal = cells->areas[left_local_cell_id];
        PetscReal arear = cells->areas[right_local_cell_id];

        PetscReal                 cnum              = amax_vec_int[e] * edge_len / fmin(areal, arear) * dt;
        CourantNumberDiagnostics *courant_num_diags = &interior_flux_op->diagnostics->courant_number;
        if (cnum > courant_num_diags->max_courant_num) {
          courant_num_diags->max_courant_num = cnum;
          courant_num_diags->global_edge_id  = edges->global_ids[e];
          if (areal < arear) courant_num_diags->global_cell_id = cells->global_ids[left_local_cell_id];
          else courant_num_diags->global_cell_id = cells->global_ids[right_local_cell_id];
        }

        for (PetscInt i_dof = 0; i_dof < n_dof; i_dof++) {
          if (cells->is_owned[left_local_cell_id]) {
            PetscInt left_owned_cell_id = cells->local_to_owned[left_local_cell_id];
            f_ptr[n_dof * left_owned_cell_id + i_dof] += flux_vec_int[n_dof * e + i_dof] * (-edge_len / areal);
          }

          if (cells->is_owned[right_local_cell_id]) {
            PetscInt right_owned_cell_id = cells->local_to_owned[right_local_cell_id];
            f_ptr[n_dof * right_owned_cell_id + i_dof] += flux_vec_int[n_dof * e + i_dof] * (edge_len / arear);
          }
        }
      }
    }
  }

  // Restore vectors
  PetscCall(VecRestoreArray(u_local, &u_ptr));
  PetscCall(VecRestoreArray(f_global, &f_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Deallocate memory
/// @param context a TracerInteriorFluxOperator struct
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode DestroyTracerInteriorFlux(void *context) {
  PetscFunctionBegin;

  TracerInteriorFluxOperator *interior_flux_op = context;

  DestroyTracerRiemannStateData(interior_flux_op->left_states);
  DestroyTracerRiemannStateData(interior_flux_op->right_states);
  DestroyTracerRiemannEdgeData(interior_flux_op->edges);

  PetscCall(PetscFree(interior_flux_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates an operator for computing fluxes through the interior edges.
/// @param [in]  mesh        mesh defining the computational domain of the operator
/// @param [in]  config      RDycore's configuration
/// @param [in]  diagnostics an OperatorDiagnostics struct
/// @param [out] petsc_op    a PetscOperator struct that is created and returned
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreatePetscTracerInteriorFluxOperator(RDyMesh *mesh, const RDyConfig config, OperatorDiagnostics *diagnostics,
                                                     PetscOperator *petsc_op) {
  PetscFunctionBegin;

  PetscInt num_flow_comp    = 3;  // NOTE: SWE assumed!
  PetscInt num_tracers_comp = config.physics.sediment.num_classes;

  TracerInteriorFluxOperator *interior_flux_op;
  PetscCall(PetscCalloc1(1, &interior_flux_op));
  *interior_flux_op = (TracerInteriorFluxOperator){
      .mesh        = mesh,
      .diagnostics = diagnostics,
      .tiny_h      = config.physics.flow.tiny_h,
      .h_anuga_regular = config.physics.flow.h_anuga_regular,
  };

  // allocate left/right/edge Riemann data structures
  PetscCall(CreateTracerRiemannStateData(mesh->num_internal_edges, num_flow_comp, num_tracers_comp, &interior_flux_op->left_states));
  PetscCall(CreateTracerRiemannStateData(mesh->num_internal_edges, num_flow_comp, num_tracers_comp, &interior_flux_op->right_states));
  PetscCall(CreateTracerRiemannEdgeData(mesh->num_internal_edges, num_flow_comp, num_tracers_comp, &interior_flux_op->edges));

  // copy mesh geometry data into place
  RDyEdges *edges = &mesh->edges;
  for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
    PetscInt edge_id       = edges->internal_edge_ids[e];
    PetscInt right_cell_id = edges->cell_ids[2 * edge_id + 1];

    if (right_cell_id != -1) {
      interior_flux_op->edges.cn[e] = edges->cn[edge_id];
      interior_flux_op->edges.sn[e] = edges->sn[edge_id];
    }
  }

  // create the interior operator
  PetscCall(PetscOperatorCreate(interior_flux_op, ApplyTracerInteriorFlux, DestroyTracerInteriorFlux, petsc_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

//------------------------
// Boundary Flux Operator
//------------------------

typedef struct {
  RDyNumericsRiemann     riemann;             // riemann solver type
  RDyMesh               *mesh;                // domain mesh
  RDyBoundary            boundary;            // boundary associated with this sub-operator
  RDyCondition           boundary_condition;  // boundary condition associated with this sub-operator
  Vec                    boundary_values;     // Dirichlet boundary values vector
  Vec                    boundary_fluxes;     // boundary flux values vector
  OperatorDiagnostics   *diagnostics;         // courant number, boundary fluxes
  PetscReal              tiny_h;              // minimum water height for wet conditions
  PetscReal              h_anuga_regular;
  TracerRiemannStateData left_states;
  TracerRiemannStateData right_states;
  TracerRiemannEdgeData  edges;
  PetscReal             *cosines, *sines;  // cosine and sine of the angle between the edge and y-axis
  PetscReal             *a_max;            // maximum courant number
} TracerBoundaryFluxOperator;

/// @brief Sets values for the cells on the right of an edge, datar, based on values on the left of the
///        edge, datal, and edge geometeric attributes for a reflective boundary condition.
/// @param [in] mesh      a RDyMesh struct for the mesh
/// @param [in] boundary  a RDyBoundary struct for the boundary
/// @param [in] datal     a TracerRiemannStateData that stores values for cells left of edges
/// @param [out] datar    a TracerRiemannStateData that stores values for cells right of edges
/// @param [in] data_edge a TracerRiemannEdgeData that has geometeric attributes about edges
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode ApplyTracerReflectingBC(RDyMesh *mesh, RDyBoundary boundary, TracerRiemannStateData *datal, TracerRiemannStateData *datar,
                                              TracerRiemannEdgeData *data_edge) {
  PetscFunctionBeginUser;

  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  PetscReal *sn_vec_bnd = data_edge->sn;
  PetscReal *cn_vec_bnd = data_edge->cn;

  PetscInt num_tracers_comp = datal->num_tracers_comp;

  // compute h/u/v for right cells
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt edge_id            = boundary.edge_ids[e];
    PetscInt left_local_cell_id = edges->cell_ids[2 * edge_id];

    if (cells->is_owned[left_local_cell_id]) {
      datar->h[e] = datal->h[e];

      PetscReal dum1 = Square(sn_vec_bnd[e]) - Square(cn_vec_bnd[e]);
      PetscReal dum2 = 2.0 * sn_vec_bnd[e] * cn_vec_bnd[e];

      datar->u[e] = datal->u[e] * dum1 - datal->v[e] * dum2;
      datar->v[e] = -datal->u[e] * dum2 - datal->v[e] * dum1;

      for (PetscInt s = 0; s < num_tracers_comp; s++) {
        datar->ci[e * num_tracers_comp + s] = datal->ci[e * num_tracers_comp + s];
      }
    }
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Computes the fluxes through the boundary edges of mesh locally owned and
///        adds contribution in f_global Vec.
/// @param [in] context  a TracerInteriorFluxOperator
/// @param [in] fields   a PetscOperatorFields ! FIXME: Can possibly be deleted
/// @param [in] dt       time step             ! FIXME: Can possibly be deleted
/// @param [in] u_local  a Vec containing values for locally-owned and ghost cells
/// @param [in] f_global a Vec for storing RHS contrinution from interior edges
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode ApplyTracerBoundaryFlux(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBeginUser;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  TracerBoundaryFluxOperator *boundary_flux_op = context;

  RDyBoundary  boundary           = boundary_flux_op->boundary;
  RDyCondition boundary_condition = boundary_flux_op->boundary_condition;
  Vec          boundary_values    = boundary_flux_op->boundary_values;
  Vec          boundary_fluxes    = boundary_flux_op->boundary_fluxes;

  // get pointers to vector data
  PetscScalar *u_ptr, *f_ptr, *boundary_values_ptr, *boundary_fluxes_ptr;
  PetscCall(VecGetArray(u_local, &u_ptr));
  PetscCall(VecGetArray(f_global, &f_ptr));
  PetscCall(VecGetArray(boundary_values, &boundary_values_ptr));
  PetscCall(VecGetArray(boundary_fluxes, &boundary_fluxes_ptr));

  // apply boundary conditions
  TracerRiemannStateData *datal     = &boundary_flux_op->left_states;
  TracerRiemannStateData *datar     = &boundary_flux_op->right_states;
  TracerRiemannEdgeData  *data_edge = &boundary_flux_op->edges;

  PetscInt num_flow_comp    = datal->num_flow_comp;
  PetscInt num_tracers_comp = datal->num_tracers_comp;

  PetscInt n_dof;
  PetscCall(VecGetBlockSize(u_local, &n_dof));
  PetscCheck(n_dof == num_flow_comp + num_tracers_comp, comm, PETSC_ERR_USER, "Number of dof in local vector do not match flow and tracers dof!");

  // copy the "left cell" values into the "left states"
  RDyEdges *edges = &boundary_flux_op->mesh->edges;
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt edge_id            = boundary.edge_ids[e];
    PetscInt left_local_cell_id = edges->cell_ids[2 * edge_id];
    datal->h[e]                 = u_ptr[n_dof * left_local_cell_id + 0];
    datal->hu[e]                = u_ptr[n_dof * left_local_cell_id + 1];
    datal->hv[e]                = u_ptr[n_dof * left_local_cell_id + 2];

    for (PetscInt s = 0; s < num_tracers_comp; s++) {
      datal->hci[e * num_tracers_comp + s] = u_ptr[n_dof * left_local_cell_id + 3 + s];
    }
  }

  // compute diagnostic quantities from prognostic variables
  const PetscReal tiny_h  = boundary_flux_op->tiny_h;
  const PetscReal h_anuga = boundary_flux_op->h_anuga_regular;
  PetscCall(ComputeRiemannVelocitiesAndConcentration(tiny_h, h_anuga, datal));

  // compute the "right" Riemann cell values using the boundary condition
  switch (boundary_condition.flow->type) {
    case CONDITION_DIRICHLET:
      // copy Dirichlet boundary values into the "right states"
      for (PetscInt e = 0; e < boundary.num_edges; ++e) {
        datar->h[e]  = boundary_values_ptr[n_dof * e + 0];
        datar->hu[e] = boundary_values_ptr[n_dof * e + 1];
        datar->hv[e] = boundary_values_ptr[n_dof * e + 2];
        for (PetscInt s = 0; s < num_tracers_comp; s++) {
          datar->hci[e * num_tracers_comp + s] = boundary_values_ptr[n_dof * e + 3 + s];
        }
      }
      PetscCall(ComputeRiemannVelocitiesAndConcentration(tiny_h, h_anuga, datar));
      break;
    case CONDITION_REFLECTING:
      PetscCall(ApplyTracerReflectingBC(boundary_flux_op->mesh, boundary, datal, datar, data_edge));
      break;
    case CONDITION_CRITICAL_OUTFLOW:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "CONDITION_CRITICAL_OUTFLOW not supported for tracers");
      break;
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Invalid boundary condition encountered for boundary %" PetscInt_FMT "\n", boundary.id);
  }

  // call Riemann solver
  switch (boundary_flux_op->riemann) {
    case RIEMANN_ROE:
      PetscCall(ComputeTracerRoeFlux(datal, datar, data_edge->sn, data_edge->cn, boundary_fluxes_ptr, data_edge->amax));
      break;
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Unsupported Riemann solver");
  }

  const PetscReal eps_flux = 1e-14;
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    const PetscReal Fh = (PetscReal)boundary_fluxes_ptr[n_dof * e + 0];
    for (PetscInt s = 0; s < num_tracers_comp; ++s) {
      const PetscInt  ci_off = e * num_tracers_comp + s;
      const PetscReal c_int  = (PetscReal)datal->ci[ci_off];
      const PetscReal c_bc   = (PetscReal)datar->ci[ci_off];
      PetscReal       Cup    = 0.0;
      if (Fh > eps_flux) {
        Cup = c_int;
      } else if (Fh < -eps_flux) {
        Cup = c_bc;
      }
      boundary_fluxes_ptr[n_dof * e + 3 + s] = Fh * Cup;
    }
  }

  // accumulate the flux values in f_global
  RDyCells                 *cells             = &boundary_flux_op->mesh->cells;
  CourantNumberDiagnostics *courant_num_diags = &boundary_flux_op->diagnostics->courant_number;
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt  edge_id       = boundary.edge_ids[e];
    PetscReal edge_len      = edges->lengths[edge_id];
    PetscInt  local_cell_id = edges->cell_ids[2 * edge_id];

    if (cells->is_owned[local_cell_id]) {
      PetscReal cell_area = cells->areas[local_cell_id];
      PetscReal hl        = datal->h[e];
      PetscReal hr        = datar->h[e];

      if (!(hl < tiny_h && hr < tiny_h)) {
        PetscReal cnum = data_edge->amax[e] * edge_len / cell_area * dt;
        if (cnum > courant_num_diags->max_courant_num) {
          courant_num_diags->max_courant_num = cnum;
          courant_num_diags->global_edge_id  = edges->global_ids[e];
          courant_num_diags->global_cell_id  = cells->global_ids[local_cell_id];
        }

        PetscInt owned_cell_id = cells->local_to_owned[local_cell_id];
        for (PetscInt i_dof = 0; i_dof < n_dof; i_dof++) {
          f_ptr[n_dof * owned_cell_id + i_dof] += boundary_fluxes_ptr[n_dof * e + i_dof] * (-edge_len / cell_area);
        }
      }
    }
  }

  // restore vectors
  PetscCall(VecRestoreArray(u_local, &u_ptr));
  PetscCall(VecRestoreArray(f_global, &f_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Deallocate memory
/// @param context a DestroyTracerBoundaryFlux struct
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode DestroyTracerBoundaryFlux(void *context) {
  PetscFunctionBegin;

  TracerBoundaryFluxOperator *boundary_flux_op = context;

  DestroyTracerRiemannStateData(boundary_flux_op->left_states);
  DestroyTracerRiemannStateData(boundary_flux_op->right_states);
  DestroyTracerRiemannEdgeData(boundary_flux_op->edges);

  PetscCall(PetscFree(boundary_flux_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates an operator for computing fluxes through boundary edges.
/// @param [in]  mesh               mesh defining the computational domain of the operator
/// @param [in]  config             RDycore's configuration
/// @param [in]  boundary           a RDyBoundary struct for the boundary edges
/// @param [in]  boundary_condition a RDyCondition struct for all boundary conditions
/// @param [in]  boundary_values    a Vec containing values for the boundary conditions
/// @param [in]  boundary_fluxes    a Vec to accumulating boundary fluxes
/// @param [in]  diagnostics        an OperatorDiagnostics struct
/// @param [out] petsc_op           a PetscOperator struct that is created and returned
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreatePetscTracerBoundaryFluxOperator(RDyMesh *mesh, const RDyConfig config, RDyBoundary boundary, RDyCondition boundary_condition,
                                                     Vec boundary_values, Vec boundary_fluxes, OperatorDiagnostics *diagnostics,
                                                     PetscOperator *petsc_op) {
  PetscFunctionBegin;

  PetscInt num_flow_comp    = 3;  // NOTE: SWE assumed!
  PetscInt num_tracers_comp = config.physics.sediment.num_classes;

  TracerBoundaryFluxOperator *boundary_flux_op;
  PetscCall(PetscCalloc1(1, &boundary_flux_op));
  *boundary_flux_op = (TracerBoundaryFluxOperator){
      .mesh               = mesh,
      .boundary           = boundary,
      .boundary_condition = boundary_condition,
      .boundary_values    = boundary_values,
      .boundary_fluxes    = boundary_fluxes,
      .diagnostics        = diagnostics,
      .tiny_h             = config.physics.flow.tiny_h,
      .h_anuga_regular    = config.physics.flow.h_anuga_regular,
  };

  // allocate left/right/edge Riemann data structures
  PetscCall(CreateTracerRiemannStateData(boundary.num_edges, num_flow_comp, num_tracers_comp, &boundary_flux_op->left_states));
  PetscCall(CreateTracerRiemannStateData(boundary.num_edges, num_flow_comp, num_tracers_comp, &boundary_flux_op->right_states));
  PetscCall(CreateTracerRiemannEdgeData(boundary.num_edges, num_flow_comp, num_tracers_comp, &boundary_flux_op->edges));

  // copy mesh geometry data into place
  RDyEdges *edges = &mesh->edges;
  for (PetscInt e = 0; e < boundary.num_edges; ++e) {
    PetscInt edge_id              = boundary.edge_ids[e];
    boundary_flux_op->edges.cn[e] = edges->cn[edge_id];
    boundary_flux_op->edges.sn[e] = edges->sn[edge_id];
  }

  // create the boundary operator
  PetscCall(PetscOperatorCreate(boundary_flux_op, ApplyTracerBoundaryFlux, DestroyTracerBoundaryFlux, petsc_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

//-----------------
// Source Operator
//-----------------

typedef struct {
  RDyMesh  *mesh;              // domain mesh
  PetscInt  num_flow_comp;     // number of flow components
  PetscInt  num_tracers_comp;  // number of tracers components
  Vec       external_sources;  // external source vector
  Vec       net_flux_by_class; // per-class sediment net flux diagnostics vector
  Vec       mannings;          // mannings coefficient vector
  PetscReal tiny_h;            // minimum water height for wet conditions
  PetscReal xq2018_threshold;  // threshold for the XQ2018's implicit time integration of source term
  RDyPhysicsSD sediment;
  TracerBedModel bed;
} TracerSourceOperator;

/// @brief Set the contribution of the source-term using the semi-implicit time integeration method
///        for the friction term in SWE.
/// @param [in] context  a TracerInteriorFluxOperator
/// @param [in] fields   a PetscOperatorFields
/// @param [in] dt       time step
/// @param [in] u_local  a Vec containing values for locally-owned and ghost cells
/// @param [in] f_global a Vec for storing RHS contrinution from interior edges
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode ApplyTracerSourceSemiImplicit(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBeginUser;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  TracerSourceOperator *source_op        = context;
  Vec                   source_vec       = source_op->external_sources;
  Vec                   net_flux_cls_vec = source_op->net_flux_by_class;
  Vec                   mannings_vec     = source_op->mannings;
  RDyMesh              *mesh             = source_op->mesh;
  RDyCells             *cells            = &mesh->cells;
  PetscReal             tiny_h           = source_op->tiny_h;
  PetscInt              num_tracers_comp = source_op->num_tracers_comp;
  TracerBedModel       *bed              = &source_op->bed;
  const PetscReal rhow             = DENSITY_OF_WATER;
  const PetscReal h_ero_min        = PetscMax(1e-1, 10.0 * tiny_h);
  const PetscReal h_sed_source_min = 1e-2;

  // access Vec data
  PetscScalar *source_ptr, *net_flux_cls_ptr, *mannings_ptr, *u_ptr, *f_ptr;
  PetscCall(VecGetArray(source_vec, &source_ptr));      // sequential vector
  PetscCall(VecGetArray(net_flux_cls_vec, &net_flux_cls_ptr));
  PetscCall(VecGetArray(mannings_vec, &mannings_ptr));  // sequential vector
  PetscCall(VecGetArray(u_local, &u_ptr));              // domain local vector (indexed by local cells)
  PetscCall(VecGetArray(f_global, &f_ptr));             // domain global vector (indexed by owned cells)

  // access previously-computed flux divergence data
  Vec flux_div;
  PetscCall(PetscOperatorFieldsGet(fields, "riemannf", &flux_div));
  PetscCheck(flux_div, comm, PETSC_ERR_USER, "No 'riemannf' field found in source operator!");
  PetscScalar *flux_div_ptr;
  PetscCall(VecGetArray(flux_div, &flux_div_ptr));  // domain global vector

  PetscInt n_dof;
  PetscCall(VecGetBlockSize(u_local, &n_dof));
  PetscReal *eroded_mass_by_class = NULL;
  PetscCall(PetscCalloc1(num_tracers_comp, &eroded_mass_by_class));

  for (PetscInt c = 0; c < mesh->num_cells; ++c) {
    if (cells->is_owned[c]) {
      PetscInt owned_cell_id = cells->local_to_owned[c];

      PetscReal h  = u_ptr[n_dof * c + 0];
      PetscReal hu = u_ptr[n_dof * c + 1];
      PetscReal hv = u_ptr[n_dof * c + 2];

      PetscReal dz_dx = cells->dz_dx[c];
      PetscReal dz_dy = cells->dz_dy[c];

      PetscReal bedx = dz_dx * GRAVITY * h;
      PetscReal bedy = dz_dy * GRAVITY * h;

      PetscReal Fsum_x = flux_div_ptr[n_dof * owned_cell_id + 1];
      PetscReal Fsum_y = flux_div_ptr[n_dof * owned_cell_id + 2];

      PetscReal tbx = 0.0, tby = 0.0;

      if (h >= tiny_h) {  // wet conditions
        PetscReal u = hu / h;
        PetscReal v = hv / h;

        // Manning's coefficient
        PetscReal N_mannings = mannings_ptr[c];

        // Cd = g n^2 h^{-1/3}, where n is Manning's coefficient
        PetscReal Cd = GRAVITY * Square(N_mannings) * PetscPowReal(h, -1.0 / 3.0);

        PetscReal velocity = PetscSqrtReal(Square(u) + Square(v));
        PetscReal tb       = Cd * velocity / h;
        PetscReal factor   = tb / (1.0 + dt * tb);

        tbx = (hu + dt * Fsum_x - dt * bedx) * factor;
        tby = (hv + dt * Fsum_y - dt * bedy) * factor;

        //PetscReal tau_b        = 0.5 * rhow * Cd * (Square(u) + Square(v));
	PetscReal tau_b        = rhow * Cd * (Square(u) + Square(v));
        ComputeTracerErodedMassByClass(bed, c, mesh->num_cells, num_tracers_comp, h, h_ero_min, tau_b, dt, eroded_mass_by_class);
        PetscReal Mtot_a      = ComputeTracerLayerTotalMass(bed, c, 0, mesh->num_cells, num_tracers_comp);
        PetscReal active_conc = ComputeTracerActiveLayerConcentration(bed, c, mesh->num_cells, num_tracers_comp);

        PetscReal deposited_mass_total = 0.0;

        for (PetscInt s = 0; s < num_tracers_comp; s++) {
          PetscReal hc      = u_ptr[n_dof * c + 3 + s];
          PetscReal ci      = hc / h;
          PetscReal ws_s    = bed->sediment->classes[s].settling_velocity;
          PetscReal tau_d_s = bed->sediment->classes[s].critical_deposition_shear_stress;

          PetscReal ei = 0.0;
          PetscReal di = 0.0;
          if (dt > 0.0) ei = eroded_mass_by_class[s] / dt;
          if (tau_d_s > 0.0 && tau_b < tau_d_s) {
            PetscReal dep_factor = 1.0 - tau_b / tau_d_s;
            di = ws_s * ci * dep_factor;
            if (di < 0.0) di = 0.0;
          }

          PetscReal net_flux = ei - di;
          if (dt > 0.0) {
            PetscReal min_net_flux = -hc / dt;
            if (net_flux < min_net_flux) net_flux = min_net_flux;
          }

          PetscInt  idx_a               = BED_INDEX(0, c, s, mesh->num_cells, num_tracers_comp);
          PetscReal deposited_mass      = PetscMax(0.0, (ei - net_flux) * dt);
          PetscReal external_sed_source = source_ptr[n_dof * owned_cell_id + 3 + s] * PetscMin(1.0, h / h_sed_source_min);

          f_ptr[n_dof * owned_cell_id + 3 + s] += net_flux + external_sed_source;
          net_flux_cls_ptr[num_tracers_comp * owned_cell_id + s] = net_flux;
          bed->bed_mass[idx_a] += deposited_mass;
          deposited_mass_total += deposited_mass;
        }

        if (deposited_mass_total > 0.0 && bed->num_bed_layers > 1) {
          PetscReal donor_conc = TracerLayerConcentration(bed, 1);
          if (Mtot_a + deposited_mass_total > 0.0) {
            active_conc = (active_conc * Mtot_a + donor_conc * deposited_mass_total) / (Mtot_a + deposited_mass_total);
          } else {
            active_conc = donor_conc;
          }
          bed->active_layer_concentration[c] = ClampTracerActiveLayerConcentration(bed, active_conc);
        }
      }

      // NOTE: we accumulate everything into the RHS vector by convention.
      f_ptr[n_dof * owned_cell_id + 0] += source_ptr[n_dof * owned_cell_id + 0];
      f_ptr[n_dof * owned_cell_id + 1] += -bedx - tbx + source_ptr[n_dof * owned_cell_id + 1];
      f_ptr[n_dof * owned_cell_id + 2] += -bedy - tby + source_ptr[n_dof * owned_cell_id + 2];
    }
  }

  // restore vectors
  PetscCall(VecRestoreArray(u_local, &u_ptr));
  PetscCall(VecRestoreArray(f_global, &f_ptr));
  PetscCall(VecRestoreArray(source_vec, &source_ptr));
  PetscCall(VecRestoreArray(net_flux_cls_vec, &net_flux_cls_ptr));
  PetscCall(VecRestoreArray(mannings_vec, &mannings_ptr));
  PetscCall(VecRestoreArray(flux_div, &flux_div_ptr));
  PetscCall(PetscFree(eroded_mass_by_class));

  PetscCall(UpdateTracerBedActiveLayer(mesh, num_tracers_comp, bed));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Deallocate memory
/// @param context a TracerSourceOperator struct
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode DestroyTracerSource(void *context) {
  PetscFunctionBegin;
  TracerSourceOperator *source_op = context;
  PetscCall(DestroyTracerBedModel(&source_op->bed));
  PetscFree(source_op);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates an operator for computing source term contribution
/// @param [in]  mesh             mesh defining the computational domain of the operator
/// @param [in]  config           RDycore's configuration
/// @param [in]  external_sources a Vec containing source values for locally-owned cells
/// @param [in]  mannings         a Vec containing Manning roughness coefficient for SWE
/// @param [out] petsc_op         a PetscOperator struct that is created and returned
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreatePetscTracerSourceOperator(RDyMesh *mesh, const RDyConfig config, Vec external_sources, Vec net_flux_by_class, Vec mannings,
                                               PetscOperator *petsc_op) {
  PetscFunctionBegin;

  PetscInt num_flow_comp    = 3;  // NOTE: SWE assumed!
  PetscInt num_tracers_comp = config.physics.sediment.num_classes;

  TracerSourceOperator *source_op;
  PetscCall(PetscCalloc1(1, &source_op));
  *source_op = (TracerSourceOperator){
      .mesh             = mesh,
      .num_flow_comp    = num_flow_comp,
      .num_tracers_comp = num_tracers_comp,
      .external_sources = external_sources,
      .net_flux_by_class = net_flux_by_class,
      .mannings         = mannings,
      .tiny_h           = config.physics.flow.tiny_h,
      .xq2018_threshold = config.physics.flow.source.xq2018_threshold,
      .sediment         = config.physics.sediment,
  };

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)external_sources, &comm));
  PetscCall(InitializeTracerBedModel(mesh, &source_op->sediment, num_tracers_comp, comm, &source_op->bed));

  switch (config.physics.flow.source.method) {
    case SOURCE_SEMI_IMPLICIT:
      PetscCall(PetscOperatorCreate(source_op, ApplyTracerSourceSemiImplicit, DestroyTracerSource, petsc_op));
      break;
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Only semi_implicit and implicit_xq2018 are supported in the PETSc version");
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

//----------------------------------
// Interior Flux HR Operator (Tracer)
//----------------------------------

typedef struct {
  RDyNumericsRiemann     riemann;           // riemann solver type
  RDyMesh               *mesh;              // domain mesh
  PetscReal              tiny_h;            // minimum water height for wet conditions
  PetscReal              h_anuga_regular;   // ANUGA height parameter for velocity regularization
  PetscReal             *zc;                // vertex-averaged bed elevation per cell (local indexing)
  PetscInt               num_flow_comp;     // number of flow components (3)
  PetscInt               num_tracers_comp;  // number of tracer components
  TracerRiemannStateData left_states;       // reconstructed "left" states
  TracerRiemannStateData right_states;      // reconstructed "right" states
  TracerRiemannEdgeData  edges;             // riemann fluxes on interior edges
  OperatorDiagnostics   *diagnostics;       // courant number, etc
} TracerInteriorFluxHROperator;

static PetscErrorCode ApplyTracerInteriorFluxHR(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBegin;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  TracerInteriorFluxHROperator *op = context;

  RDyMesh  *mesh  = op->mesh;
  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  PetscScalar *u_ptr, *f_ptr;
  PetscCall(VecGetArray(u_local, &u_ptr));
  PetscCall(VecGetArray(f_global, &f_ptr));

  TracerRiemannStateData *datal        = &op->left_states;
  TracerRiemannStateData *datar        = &op->right_states;
  TracerRiemannEdgeData  *data_edge    = &op->edges;
  PetscReal              *sn_vec_int   = data_edge->sn;
  PetscReal              *cn_vec_int   = data_edge->cn;
  PetscReal              *amax_vec_int = data_edge->amax;
  PetscReal              *flux_vec_int = data_edge->fluxes;

  PetscInt num_flow_comp    = datal->num_flow_comp;
  PetscInt num_tracers_comp = datal->num_tracers_comp;

  PetscInt n_dof;
  PetscCall(VecGetBlockSize(u_local, &n_dof));
  PetscCheck(n_dof == num_flow_comp + num_tracers_comp, comm, PETSC_ERR_USER,
             "Mismatch in number of dof in local vector (%" PetscInt_FMT ") and flow + tracers (%" PetscInt_FMT ")", n_dof,
             num_flow_comp + num_tracers_comp);

  const PetscReal  tiny_h  = op->tiny_h;
  const PetscReal  h_anuga = op->h_anuga_regular;
  const PetscReal *zc      = op->zc;

  // For each internal edge, perform hydrostatic reconstruction and compute the
  // Roe flux on the reconstructed states.
  for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
    PetscInt edge_id = edges->internal_edge_ids[e];
    PetscInt l       = edges->cell_ids[2 * edge_id];
    PetscInt r       = edges->cell_ids[2 * edge_id + 1];

    if (r == -1) continue;

    PetscReal h_L  = u_ptr[n_dof * l + 0];
    PetscReal hu_L = u_ptr[n_dof * l + 1];
    PetscReal hv_L = u_ptr[n_dof * l + 2];
    PetscReal h_R  = u_ptr[n_dof * r + 0];
    PetscReal hu_R = u_ptr[n_dof * r + 1];
    PetscReal hv_R = u_ptr[n_dof * r + 2];

    // hydrostatic reconstruction
    PetscReal zc_L   = zc[l];
    PetscReal zc_R   = zc[r];
    PetscReal eta_L  = h_L + zc_L;
    PetscReal eta_R  = h_R + zc_R;
    PetscReal z_max  = fmax(zc_L, zc_R);
    PetscReal hL_rec = fmax(0.0, eta_L - z_max);
    PetscReal hR_rec = fmax(0.0, eta_R - z_max);

    // preserve velocities using ANUGA regularization
    PetscReal denom_L = Square(h_L) + Square(h_anuga);
    PetscReal denom_R = Square(h_R) + Square(h_anuga);
    PetscReal uL      = (h_L > tiny_h) ? hu_L * h_L / denom_L : 0.0;
    PetscReal vL      = (h_L > tiny_h) ? hv_L * h_L / denom_L : 0.0;
    PetscReal uR      = (h_R > tiny_h) ? hu_R * h_R / denom_R : 0.0;
    PetscReal vR      = (h_R > tiny_h) ? hv_R * h_R / denom_R : 0.0;

    // set reconstructed flow states
    datal->h[e] = hL_rec;
    datal->u[e] = uL;
    datal->v[e] = vL;
    datar->h[e] = hR_rec;
    datar->u[e] = uR;
    datar->v[e] = vR;

    // preserve concentrations: ci_rec = ci (original), hci_rec = ci * h_rec
    for (PetscInt s = 0; s < num_tracers_comp; s++) {
      PetscReal hci_L = u_ptr[n_dof * l + 3 + s];
      PetscReal hci_R = u_ptr[n_dof * r + 3 + s];
      PetscReal ci_L  = (h_L > tiny_h) ? hci_L / h_L : 0.0;
      PetscReal ci_R  = (h_R > tiny_h) ? hci_R / h_R : 0.0;

      datal->hci[e * num_tracers_comp + s] = hL_rec * ci_L;
      datar->hci[e * num_tracers_comp + s] = hR_rec * ci_R;
      datal->ci[e * num_tracers_comp + s]  = ci_L;
      datar->ci[e * num_tracers_comp + s]  = ci_R;
    }
  }

  // call Riemann solver on reconstructed states
  switch (op->riemann) {
    case RIEMANN_ROE:
      PetscCall(ComputeTracerRoeFlux(datal, datar, sn_vec_int, cn_vec_int, flux_vec_int, amax_vec_int));
      break;
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Unsupported Riemann solver");
  }

  // accumulate fluxes + hydrostatic pressure correction
  for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
    PetscInt edge_id = edges->internal_edge_ids[e];
    PetscInt l       = edges->cell_ids[2 * edge_id];
    PetscInt r       = edges->cell_ids[2 * edge_id + 1];

    if (r == -1) continue;

    PetscReal h_L = u_ptr[n_dof * l + 0];
    PetscReal h_R = u_ptr[n_dof * r + 0];

    if (!(h_R < tiny_h && h_L < tiny_h)) {
      PetscReal edge_len = edges->lengths[edge_id];
      PetscReal areal    = cells->areas[l];
      PetscReal arear    = cells->areas[r];

      // hydrostatic reconstruction (recompute for correction)
      PetscReal zc_L   = zc[l];
      PetscReal zc_R   = zc[r];
      PetscReal eta_L  = h_L + zc_L;
      PetscReal eta_R  = h_R + zc_R;
      PetscReal z_max  = fmax(zc_L, zc_R);
      PetscReal hL_rec = fmax(0.0, eta_L - z_max);
      PetscReal hR_rec = fmax(0.0, eta_R - z_max);

      PetscReal flux_scale_l = -edge_len / areal;
      PetscReal flux_scale_r = edge_len / arear;

      if (hL_rec > tiny_h || hR_rec > tiny_h) {
        // Courant number diagnostic
        PetscReal                 cnum              = amax_vec_int[e] * edge_len / fmin(areal, arear) * dt;
        CourantNumberDiagnostics *courant_num_diags = &op->diagnostics->courant_number;
        if (cnum > courant_num_diags->max_courant_num) {
          courant_num_diags->max_courant_num = cnum;
          courant_num_diags->global_edge_id  = edges->global_ids[e];
          if (areal < arear) courant_num_diags->global_cell_id = cells->global_ids[l];
          else courant_num_diags->global_cell_id = cells->global_ids[r];
        }

        // accumulate Roe fluxes
        for (PetscInt i_dof = 0; i_dof < n_dof; i_dof++) {
          if (cells->is_owned[l]) {
            PetscInt lo = cells->local_to_owned[l];
            f_ptr[n_dof * lo + i_dof] += flux_vec_int[n_dof * e + i_dof] * flux_scale_l;
          }
          if (cells->is_owned[r]) {
            PetscInt ro = cells->local_to_owned[r];
            f_ptr[n_dof * ro + i_dof] += flux_vec_int[n_dof * e + i_dof] * flux_scale_r;
          }
        }
      }

      // hydrostatic pressure correction (momentum components only)
      PetscReal corr_L = 0.5 * GRAVITY * (Square(h_L) - Square(hL_rec));
      PetscReal corr_R = 0.5 * GRAVITY * (Square(h_R) - Square(hR_rec));
      PetscReal cn     = cn_vec_int[e];
      PetscReal sn     = sn_vec_int[e];

      if (cells->is_owned[l]) {
        PetscInt lo = cells->local_to_owned[l];
        f_ptr[n_dof * lo + 1] += corr_L * cn * flux_scale_l;
        f_ptr[n_dof * lo + 2] += corr_L * sn * flux_scale_l;
      }
      if (cells->is_owned[r]) {
        PetscInt ro = cells->local_to_owned[r];
        f_ptr[n_dof * ro + 1] += corr_R * cn * flux_scale_r;
        f_ptr[n_dof * ro + 2] += corr_R * sn * flux_scale_r;
      }
    }
  }

  PetscCall(VecRestoreArray(u_local, &u_ptr));
  PetscCall(VecRestoreArray(f_global, &f_ptr));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyTracerInteriorFluxHR(void *context) {
  PetscFunctionBegin;
  TracerInteriorFluxHROperator *op = context;
  DestroyTracerRiemannStateData(op->left_states);
  DestroyTracerRiemannStateData(op->right_states);
  DestroyTracerRiemannEdgeData(op->edges);
  PetscCall(PetscFree(op->zc));
  PetscCall(PetscFree(op));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates a PetscOperator that computes interior fluxes with hydrostatic
/// reconstruction for shallow water equations + tracers.
PetscErrorCode CreatePetscTracerInteriorFluxHROperator(RDyMesh *mesh, const RDyConfig config, OperatorDiagnostics *diagnostics,
                                                       PetscOperator *petsc_op) {
  PetscFunctionBegin;

  PetscInt num_flow_comp    = 3;  // NOTE: SWE assumed!
  PetscInt num_tracers_comp = config.physics.sediment.num_classes;

  TracerInteriorFluxHROperator *op;
  PetscCall(PetscCalloc1(1, &op));
  *op = (TracerInteriorFluxHROperator){
      .riemann          = config.numerics.riemann,
      .mesh             = mesh,
      .diagnostics      = diagnostics,
      .tiny_h           = config.physics.flow.tiny_h,
      .h_anuga_regular  = config.physics.flow.h_anuga_regular,
      .num_flow_comp    = num_flow_comp,
      .num_tracers_comp = num_tracers_comp,
  };

  // allocate Riemann data structures
  PetscCall(CreateTracerRiemannStateData(mesh->num_internal_edges, num_flow_comp, num_tracers_comp, &op->left_states));
  PetscCall(CreateTracerRiemannStateData(mesh->num_internal_edges, num_flow_comp, num_tracers_comp, &op->right_states));
  PetscCall(CreateTracerRiemannEdgeData(mesh->num_internal_edges, num_flow_comp, num_tracers_comp, &op->edges));

  // copy edge geometry
  RDyEdges *edges = &mesh->edges;
  for (PetscInt e = 0; e < mesh->num_internal_edges; e++) {
    PetscInt edge_id       = edges->internal_edge_ids[e];
    PetscInt right_cell_id = edges->cell_ids[2 * edge_id + 1];
    if (right_cell_id != -1) {
      op->edges.cn[e] = edges->cn[edge_id];
      op->edges.sn[e] = edges->sn[edge_id];
    }
  }

  // compute vertex-averaged bed elevation for each cell
  RDyCells    *cells    = &mesh->cells;
  RDyVertices *vertices = &mesh->vertices;
  PetscCall(PetscCalloc1(mesh->num_cells, &op->zc));
  for (PetscInt c = 0; c < mesh->num_cells; c++) {
    PetscReal z_sum = 0.0;
    for (PetscInt v = cells->vertex_offsets[c]; v < cells->vertex_offsets[c + 1]; v++) {
      z_sum += vertices->points[cells->vertex_ids[v]].X[2];
    }
    op->zc[c] = z_sum / (PetscReal)cells->num_vertices[c];
  }

  PetscCall(PetscOperatorCreate(op, ApplyTracerInteriorFluxHR, DestroyTracerInteriorFluxHR, petsc_op));

  PetscFunctionReturn(PETSC_SUCCESS);
}

//-------------------------------
// Source HR Operator (Tracer)
//-------------------------------

typedef struct {
  RDyMesh  *mesh;              // domain mesh
  PetscInt  num_flow_comp;     // number of flow components
  PetscInt  num_tracers_comp;  // number of tracers components
  Vec       external_sources;  // external source vector
  Vec       net_flux_by_class; // per-class sediment net flux diagnostics vector
  Vec       mannings;          // mannings coefficient vector
  PetscReal tiny_h;            // minimum water height for wet conditions
  PetscReal xq2018_threshold;  // threshold for XQ2018
  RDyPhysicsSD sediment;
  TracerBedModel bed;
} TracerSourceHROperator;

static PetscErrorCode ApplyTracerSourceHRSemiImplicit(void *context, PetscOperatorFields fields, PetscReal dt, Vec u_local, Vec f_global) {
  PetscFunctionBeginUser;

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)u_local, &comm));

  TracerSourceHROperator *source_op        = context;
  Vec                     source_vec       = source_op->external_sources;
  Vec                     net_flux_cls_vec = source_op->net_flux_by_class;
  Vec                     mannings_vec     = source_op->mannings;
  RDyMesh                *mesh             = source_op->mesh;
  RDyCells               *cells            = &mesh->cells;
  PetscReal               tiny_h           = source_op->tiny_h;
  PetscInt                num_tracers_comp = source_op->num_tracers_comp;
  TracerBedModel         *bed              = &source_op->bed;
  const PetscReal rhow             = DENSITY_OF_WATER;
  const PetscReal h_ero_min        = PetscMax(1e-1, 10.0 * tiny_h);
  const PetscReal h_sed_source_min = 1e-2;

  PetscScalar *source_ptr, *net_flux_cls_ptr, *mannings_ptr, *u_ptr, *f_ptr;
  PetscCall(VecGetArray(source_vec, &source_ptr));
  PetscCall(VecGetArray(net_flux_cls_vec, &net_flux_cls_ptr));
  PetscCall(VecGetArray(mannings_vec, &mannings_ptr));
  PetscCall(VecGetArray(u_local, &u_ptr));
  PetscCall(VecGetArray(f_global, &f_ptr));

  Vec flux_div;
  PetscCall(PetscOperatorFieldsGet(fields, "riemannf", &flux_div));
  PetscCheck(flux_div, comm, PETSC_ERR_USER, "No 'riemannf' field found in source operator!");
  PetscScalar *flux_div_ptr;
  PetscCall(VecGetArray(flux_div, &flux_div_ptr));

  PetscInt n_dof;
  PetscCall(VecGetBlockSize(u_local, &n_dof));
  PetscReal *eroded_mass_by_class = NULL;
  PetscCall(PetscCalloc1(num_tracers_comp, &eroded_mass_by_class));

  for (PetscInt c = 0; c < mesh->num_cells; ++c) {
    if (cells->is_owned[c]) {
      PetscInt owned_cell_id = cells->local_to_owned[c];

      PetscReal h  = u_ptr[n_dof * c + 0];
      PetscReal hu = u_ptr[n_dof * c + 1];
      PetscReal hv = u_ptr[n_dof * c + 2];

      // bed slope terms are zero — already accounted for by HR flux correction
      PetscReal bedx = 0.0;
      PetscReal bedy = 0.0;

      PetscReal Fsum_x = flux_div_ptr[n_dof * owned_cell_id + 1];
      PetscReal Fsum_y = flux_div_ptr[n_dof * owned_cell_id + 2];

      PetscReal tbx = 0.0, tby = 0.0;

      if (h >= tiny_h) {
        PetscReal u = hu / h;
        PetscReal v = hv / h;

        PetscReal N_mannings = mannings_ptr[c];
        PetscReal Cd         = GRAVITY * Square(N_mannings) * PetscPowReal(h, -1.0 / 3.0);
        PetscReal velocity   = PetscSqrtReal(Square(u) + Square(v));
        PetscReal tb         = Cd * velocity / h;
        PetscReal factor     = tb / (1.0 + dt * tb);

        tbx = (hu + dt * Fsum_x - dt * bedx) * factor;
        tby = (hv + dt * Fsum_y - dt * bedy) * factor;

        //PetscReal tau_b        = 0.5 * rhow * Cd * (Square(u) + Square(v));
	PetscReal tau_b        = rhow * Cd * (Square(u) + Square(v));
        ComputeTracerErodedMassByClass(bed, c, mesh->num_cells, num_tracers_comp, h, h_ero_min, tau_b, dt, eroded_mass_by_class);
        PetscReal Mtot_a      = ComputeTracerLayerTotalMass(bed, c, 0, mesh->num_cells, num_tracers_comp);
        PetscReal active_conc = ComputeTracerActiveLayerConcentration(bed, c, mesh->num_cells, num_tracers_comp);

        PetscReal deposited_mass_total = 0.0;

        for (PetscInt s = 0; s < num_tracers_comp; s++) {
          PetscReal hc      = u_ptr[n_dof * c + 3 + s];
          PetscReal ci      = hc / h;
          PetscReal ws_s    = bed->sediment->classes[s].settling_velocity;
          PetscReal tau_d_s = bed->sediment->classes[s].critical_deposition_shear_stress;

          PetscReal ei = 0.0;
          PetscReal di = 0.0;
          if (dt > 0.0) ei = eroded_mass_by_class[s] / dt;
          if (tau_d_s > 0.0 && tau_b < tau_d_s) {
            PetscReal dep_factor = 1.0 - tau_b / tau_d_s;
            di = ws_s * ci * dep_factor;
            if (di < 0.0) di = 0.0;
          }

          PetscReal net_flux = ei - di;
          if (dt > 0.0) {
            PetscReal min_net_flux = -hc / dt;
            if (net_flux < min_net_flux) net_flux = min_net_flux;
          }

          PetscInt  idx_a               = BED_INDEX(0, c, s, mesh->num_cells, num_tracers_comp);
          PetscReal deposited_mass      = PetscMax(0.0, (ei - net_flux) * dt);
          PetscReal external_sed_source = source_ptr[n_dof * owned_cell_id + 3 + s] * PetscMin(1.0, h / h_sed_source_min);

          f_ptr[n_dof * owned_cell_id + 3 + s] += net_flux + external_sed_source;
          net_flux_cls_ptr[num_tracers_comp * owned_cell_id + s] = net_flux;
          bed->bed_mass[idx_a] += deposited_mass;
          deposited_mass_total += deposited_mass;
        }

        if (deposited_mass_total > 0.0 && bed->num_bed_layers > 1) {
          PetscReal donor_conc = TracerLayerConcentration(bed, 1);
          if (Mtot_a + deposited_mass_total > 0.0) {
            active_conc = (active_conc * Mtot_a + donor_conc * deposited_mass_total) / (Mtot_a + deposited_mass_total);
          } else {
            active_conc = donor_conc;
          }
          bed->active_layer_concentration[c] = ClampTracerActiveLayerConcentration(bed, active_conc);
        }
      }

      f_ptr[n_dof * owned_cell_id + 0] += source_ptr[n_dof * owned_cell_id + 0];
      f_ptr[n_dof * owned_cell_id + 1] += -bedx - tbx + source_ptr[n_dof * owned_cell_id + 1];
      f_ptr[n_dof * owned_cell_id + 2] += -bedy - tby + source_ptr[n_dof * owned_cell_id + 2];
    }
  }

  PetscCall(VecRestoreArray(u_local, &u_ptr));
  PetscCall(VecRestoreArray(f_global, &f_ptr));
  PetscCall(VecRestoreArray(source_vec, &source_ptr));
  PetscCall(VecRestoreArray(net_flux_cls_vec, &net_flux_cls_ptr));
  PetscCall(VecRestoreArray(mannings_vec, &mannings_ptr));
  PetscCall(VecRestoreArray(flux_div, &flux_div_ptr));
  PetscCall(PetscFree(eroded_mass_by_class));

  PetscCall(UpdateTracerBedActiveLayer(mesh, num_tracers_comp, bed));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyTracerSourceHR(void *context) {
  PetscFunctionBegin;
  TracerSourceHROperator *source_op = context;
  PetscCall(DestroyTracerBedModel(&source_op->bed));
  PetscFree(source_op);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/// Creates a PetscOperator that computes source terms for HR well-balanced
/// shallow water equations + tracers (bed slope = 0, friction + erosion/deposition).
PetscErrorCode CreatePetscTracerSourceHROperator(RDyMesh *mesh, const RDyConfig config, Vec external_sources, Vec net_flux_by_class,
                                                 Vec mannings, PetscOperator *petsc_op) {
  PetscFunctionBegin;

  PetscInt num_flow_comp    = 3;  // NOTE: SWE assumed!
  PetscInt num_tracers_comp = config.physics.sediment.num_classes;

  TracerSourceHROperator *source_op;
  PetscCall(PetscCalloc1(1, &source_op));
  *source_op = (TracerSourceHROperator){
      .mesh             = mesh,
      .num_flow_comp    = num_flow_comp,
      .num_tracers_comp = num_tracers_comp,
      .external_sources = external_sources,
      .net_flux_by_class = net_flux_by_class,
      .mannings         = mannings,
      .tiny_h           = config.physics.flow.tiny_h,
      .xq2018_threshold = config.physics.flow.source.xq2018_threshold,
      .sediment         = config.physics.sediment,
  };

  MPI_Comm comm;
  PetscCall(PetscObjectGetComm((PetscObject)external_sources, &comm));
  PetscCall(InitializeTracerBedModel(mesh, &source_op->sediment, num_tracers_comp, comm, &source_op->bed));

  switch (config.physics.flow.source.method) {
    case SOURCE_SEMI_IMPLICIT:
      PetscCall(PetscOperatorCreate(source_op, ApplyTracerSourceHRSemiImplicit, DestroyTracerSourceHR, petsc_op));
      break;
    default:
      PetscCheck(PETSC_FALSE, comm, PETSC_ERR_USER, "Only semi_implicit is supported for tracer HR in the PETSc version");
      break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
