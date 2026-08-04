#include "fvCFD.H"
#include "physicoChemicalConstants.H"
#include "zeroGradientFvPatchFields.H"

#ifndef UGKWP_USE_CUDA
#error GPU-Riemann-GKP must be installed with UGKWP_USE_CUDA; run ./install.sh
#endif

#ifdef UGKWP_USE_CUDA
#include "gpu/GpuResidentStrict.H"
#include "gpu/GpuBoundarySchedule.H"
#include "autoPtr.H"
#endif

using namespace Foam;


namespace
{
#ifdef UGKWP_USE_CUDA
inline bool finiteScalar(const scalar x)
{
    return (x == x) && (mag(x) < GREAT);
}
#else
inline bool finiteScalar(const scalar x)
{
    return (x == x) && (mag(x) < GREAT);
}
#endif
}

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createFields.H"

    const bool gpuResidentStrict =
        ugkwpProps.lookupOrDefault<bool>("gpuResidentStrict", false);
    const bool gpuResidentPureGasOnly =
        ugkwpProps.lookupOrDefault<bool>("gpuResidentPureGasOnly", false);
    const Switch gpuResidentDynamicInlet =
        ugkwpProps.lookupOrDefault<Switch>
        (
            "gpuResidentDynamicInlet",
            Switch(false)
        );
    if (gpuResidentStrict)
    {
#ifndef UGKWP_USE_CUDA
        FatalErrorInFunction
            << "ugkwpProperties: gpuResidentStrict=true requires the CUDA "
            << "solver build. CPU/OpenFOAM is only allowed to read the case, "
            << "construct the mesh, read dictionaries/initial fields, write "
            << "host mirrors, restart, and log."
            << exit(FatalError);
#else
        const bool forbiddenPartialOffload =
            ugkwpProps.lookupOrDefault<bool>("useCudaGas", false)
         || ugkwpProps.lookupOrDefault<bool>("useCudaGasFused", false)
         || ugkwpProps.lookupOrDefault<bool>("useCudaSolid", false)
         || ugkwpProps.lookupOrDefault<bool>("useCudaParticleMoments", false)
         || ugkwpProps.lookupOrDefault<bool>("useCudaParticleMomentsBinned", false)
         || ugkwpProps.lookupOrDefault<bool>("useCudaParticleAdvanceInterior", false)
         || ugkwpProps.lookupOrDefault<bool>("useCudaCollisionThermalizer", false)
         || ugkwpProps.lookupOrDefault<bool>("useCudaCoupling", false)
         || ugkwpProps.lookupOrDefault<bool>("useGpuRuntimeSkeleton", false)
         || ugkwpProps.lookupOrDefault<bool>("useGpuRuntimePersistent", false)
         || ugkwpProps.lookupOrDefault<bool>("cudaBenchmarkBatchMode", false);

        if (forbiddenPartialOffload)
        {
            FatalErrorInFunction
                << "gpuResidentStrict=true rejects the legacy partial CUDA "
                << "offload switches. The strict path owns all active state "
                << "in one GPU resident driver; do not combine it with "
                << "useCudaGas/useCudaSolid/useCudaParticle*"
                << "/useCudaCoupling/useGpuRuntime*/cudaBenchmarkBatchMode."
                << exit(FatalError);
        }

        if (ugkwpProps.lookupOrDefault<bool>("useCudaRepresentativeMerge", false))
        {
            FatalErrorInFunction
                << "gpuResidentStrict=true currently rejects "
                << "useCudaRepresentativeMerge. The active strict path uses "
                << "fixed parcelMass and no representative merge."
                << exit(FatalError);
        }

        if (ugkwpProps.lookupOrDefault<bool>("cudaCollisionUseCpuRandomBridge", false))
        {
            FatalErrorInFunction
                << "gpuResidentStrict=true rejects cudaCollisionUseCpuRandomBridge. "
                << "Random state must be device resident."
                << exit(FatalError);
        }

        const scalar gpuResidentParticleCapacityInput =
            ugkwpProps.lookupOrDefault<scalar>
            (
                "gpuResidentParticleCapacity",
                1000000.0
            );
        const label gpuResidentParticleCapacity =
            label(gpuResidentParticleCapacityInput);

        if
        (
            !finiteScalar(gpuResidentParticleCapacityInput)
         || gpuResidentParticleCapacityInput < scalar(0)
         || mag
            (
                gpuResidentParticleCapacityInput
              - scalar(gpuResidentParticleCapacity)
            ) > SMALL
        )
        {
            FatalErrorInFunction
                << "ugkwpProperties: gpuResidentParticleCapacity must be a "
                << "finite non-negative integer. Current value = "
                << gpuResidentParticleCapacityInput
                << exit(FatalError);
        }

        const scalar gpuResidentSeedInput =
            ugkwpProps.lookupOrDefault<scalar>
            (
                "gpuResidentRandomSeed",
                12345.0
            );
        const unsigned long long gpuResidentSeed =
            static_cast<unsigned long long>(max(gpuResidentSeedInput, scalar(0)));

        const scalar gpuResidentMaxFaceWalkHopsInput =
            ugkwpProps.lookupOrDefault<scalar>
            (
                "gpuResidentMaxFaceWalkHops",
                8.0
            );
        const label gpuResidentMaxFaceWalkHops =
            label(gpuResidentMaxFaceWalkHopsInput);

        if
        (
            !finiteScalar(gpuResidentMaxFaceWalkHopsInput)
         || gpuResidentMaxFaceWalkHopsInput < scalar(1)
         || mag
            (
                gpuResidentMaxFaceWalkHopsInput
              - scalar(gpuResidentMaxFaceWalkHops)
            ) > SMALL
        )
        {
            FatalErrorInFunction
                << "ugkwpProperties: gpuResidentMaxFaceWalkHops must be a "
                << "finite positive integer. Current value = "
                << gpuResidentMaxFaceWalkHopsInput
                << exit(FatalError);
        }

        const scalar gpuResidentCourantUpdateIntervalInput =
            ugkwpProps.lookupOrDefault<scalar>
            (
                "gpuResidentCourantUpdateInterval",
                1000.0
            );
        const label gpuResidentCourantUpdateInterval =
            label(gpuResidentCourantUpdateIntervalInput);

        if
        (
            !finiteScalar(gpuResidentCourantUpdateIntervalInput)
         || gpuResidentCourantUpdateIntervalInput < scalar(1)
         || mag
            (
                gpuResidentCourantUpdateIntervalInput
              - scalar(gpuResidentCourantUpdateInterval)
            ) > SMALL
        )
        {
            FatalErrorInFunction
                << "ugkwpProperties: gpuResidentCourantUpdateInterval must be a "
                << "finite positive integer. Current value = "
                << gpuResidentCourantUpdateIntervalInput
                << exit(FatalError);
        }

        const scalar gpuResidentMaxDeltaTGrowth =
            ugkwpProps.lookupOrDefault<scalar>
            (
                "gpuResidentMaxDeltaTGrowth",
                scalar(1.05)
            );
        if
        (
            !finiteScalar(gpuResidentMaxDeltaTGrowth)
         || gpuResidentMaxDeltaTGrowth < scalar(1)
        )
        {
            FatalErrorInFunction
                << "ugkwpProperties: gpuResidentMaxDeltaTGrowth must be finite "
                << "and at least 1. Current value = "
                << gpuResidentMaxDeltaTGrowth
                << exit(FatalError);
        }

        const scalar epsSMinStrict =
            ugkwpProps.lookupOrDefault<scalar>("epsSMin", 1.0e-12);
        const scalar thetaMinStrict =
            ugkwpProps.lookupOrDefault<scalar>("thetaMin", SMALL);
        const scalar rhoMinStrict = rhoMinG.value();
        const scalar TgasMinStrict = TgasMinG.value();
        const scalar TpMinStrict = max(TpMin.value(), SMALL);
        const scalar TpMaxStrict = max(TpMax.value(), TpMinStrict + SMALL);
        const scalar dMinStrict =
            max(ugkwpProps.lookupOrDefault<scalar>("dMin", dS.value()), SMALL);
        const scalar dMaxStrict =
            max(ugkwpProps.lookupOrDefault<scalar>("dMax", dS.value()), dMinStrict);
        const scalar dSigmaStrict =
            max(ugkwpProps.lookupOrDefault<scalar>("dSigma", scalar(0)), scalar(0));
        const scalar gpuResidentInjectionTheta =
            max
            (
                ugkwpProps.lookupOrDefault<scalar>
                (
                    "gpuResidentInjectionTheta",
                    scalar(15000)
                ),
                scalar(0)
            );
        const scalar gpuResidentInjectionTp =
            min
            (
                max
                (
                    ugkwpProps.lookupOrDefault<scalar>
                    (
                        "gpuResidentInjectionTp",
                        TpMinStrict
                    ),
                    TpMinStrict
                ),
                TpMaxStrict
            );

        word gpuResidentInletPatch;
        word gpuResidentPressureTable;
        word gpuResidentVolumeFractionTable;
        scalar gpuResidentInletTemperature = scalar(0);
        labelList scheduledInletFaceIds;
        autoPtr<GpuBoundaryScheduleTable> pressureSchedule;
        autoPtr<GpuBoundaryScheduleTable> volumeFractionSchedule;

        if (gpuResidentDynamicInlet)
        {
            gpuResidentInletPatch =
                ugkwpProps.lookupOrDefault<word>
                (
                    "gpuResidentInletPatch",
                    "inlet"
                );
            gpuResidentPressureTable =
                ugkwpProps.lookupOrDefault<word>
                (
                    "gpuResidentPressureTable",
                    "inletPressure.table"
                );
            gpuResidentVolumeFractionTable =
                ugkwpProps.lookupOrDefault<word>
                (
                    "gpuResidentVolumeFractionTable",
                    "inletVolumeFraction.table"
                );
            gpuResidentInletTemperature =
                ugkwpProps.lookupOrDefault<scalar>
                (
                    "gpuResidentInletTemperature",
                    scalar(3600)
                );
            if
            (
                !finiteScalar(gpuResidentInletTemperature)
             || gpuResidentInletTemperature <= SMALL
            )
            {
                FatalErrorInFunction
                    << "gpuResidentInletTemperature must be positive and finite"
                    << exit(FatalError);
            }

            const label scheduledInletPatchId =
                mesh.boundaryMesh().findPatchID(gpuResidentInletPatch);
            if (scheduledInletPatchId < 0)
            {
                FatalErrorInFunction
                    << "Cannot find scheduled inlet patch "
                    << gpuResidentInletPatch << exit(FatalError);
            }
            const polyPatch& scheduledInletPatch =
                mesh.boundaryMesh()[scheduledInletPatchId];
            if (scheduledInletPatch.empty())
            {
                FatalErrorInFunction
                    << "Scheduled inlet patch is empty: "
                    << gpuResidentInletPatch << exit(FatalError);
            }
            if
            (
                !p.boundaryField()[scheduledInletPatchId].fixesValue()
             || !Tgas.boundaryField()[scheduledInletPatchId].fixesValue()
             || !epsS.boundaryField()[scheduledInletPatchId].fixesValue()
            )
            {
                FatalErrorInFunction
                    << "Scheduled inlet requires fixed-value p, T, and "
                    << "epsilonS on patch " << gpuResidentInletPatch
                    << exit(FatalError);
            }

            scheduledInletFaceIds.setSize(scheduledInletPatch.size());
            forAll(scheduledInletFaceIds, faceI)
            {
                scheduledInletFaceIds[faceI] =
                    scheduledInletPatch.start() + faceI;
            }

            pressureSchedule.reset
            (
                new GpuBoundaryScheduleTable
                (
                    runTime.path()/runTime.constant()/gpuResidentPressureTable
                )
            );
            volumeFractionSchedule.reset
            (
                new GpuBoundaryScheduleTable
                (
                    runTime.path()
                   /runTime.constant()
                   /gpuResidentVolumeFractionTable
                )
            );
            pressureSchedule->resetCursor(runTime.value());
            volumeFractionSchedule->resetCursor(runTime.value());
        }

        auto dispatchScheduledBoundary =
        [&]
        (
            auto& resident,
            const scalar scheduleTime,
            const bool force
        ) -> bool
        {
            if (!gpuResidentDynamicInlet)
            {
                return false;
            }

            const bool pressureKnotChanged =
                pressureSchedule->advanceTo(scheduleTime);
            const bool volumeFractionChanged =
                volumeFractionSchedule->advanceTo(scheduleTime);
            const bool scheduledBoundaryChanged =
                force || pressureKnotChanged || volumeFractionChanged;
            const scalar scheduledPressure =
                pressureSchedule->linearValueAt(scheduleTime);
            const scalar scheduledDensity =
                scheduledPressure
               /(Rgas.value()*gpuResidentInletTemperature);
            resident.updateScheduledInlet
            (
                scheduledInletFaceIds,
                scheduledPressure,
                scheduledDensity,
                volumeFractionSchedule->currentValue()
            );
            return scheduledBoundaryChanged;
        };

        bool adjustTimeStep = false;
        scalar maxCo = 0.5;
        scalar maxDeltaT = GREAT;

        if (gpuResidentPureGasOnly)
        {
            GpuGasResidentSolver resident;
            resident.initialise
            (
                mesh,
                rho,
                rhoU,
                rhoE,
                U,
                p,
                Tgas,
                gpuResidentMaxFaceWalkHops,
                gpuResidentSeed,
                gammaG.value(),
                Rgas.value(),
                strictOpenFoamGasBoundaries,
                muG.value(),
                PrG.value(),
                gasFluxScheme,
                gasReconstruction,
                gasLimiter,
                gasTimeIntegrator,
                gasRobustFallback,
                gasTurbulenceModel,
                gasEntropyFixCoefficient,
                lesDeltaCoeff,
                turbulentPrandtl,
                waleCw,
                smagorinskyCs,
                maxDiffusionNumber,
                rhoMinStrict,
                TgasMinStrict
            );

            dispatchScheduledBoundary(resident, runTime.value(), true);

            scalar lastMeasuredCo = scalar(0);
            label stepsSinceCourant = gpuResidentCourantUpdateInterval;

            while (runTime.run())
            {
                #include "readTimeControls.H"

                dispatchScheduledBoundary
                (
                    resident,
                    runTime.value() + runTime.deltaTValue(),
                    false
                );
                const bool courantRefreshDue =
                    adjustTimeStep
                 && stepsSinceCourant >= gpuResidentCourantUpdateInterval;
                if (courantRefreshDue)
                {
                    const scalar oldDt = runTime.deltaTValue();
                    lastMeasuredCo =
                        resident.computeGasCourant(oldDt, maxCo);
                    if (lastMeasuredCo > SMALL)
                    {
                        const scalar requestedDt =
                            oldDt*maxCo/max(lastMeasuredCo, SMALL);
                        const scalar growthLimitedDt =
                            min(requestedDt, oldDt*gpuResidentMaxDeltaTGrowth);
                        runTime.setDeltaT(min(growthLimitedDt, maxDeltaT));
                    }
                    stepsSinceCourant = 1;
                }
                else
                {
                    stepsSinceCourant = min
                    (
                        stepsSinceCourant + 1,
                        gpuResidentCourantUpdateInterval
                    );
                }

                runTime++;

                resident.advanceOneStep(runTime.deltaTValue());

                if (runTime.writeTime())
                {
                    resident.downloadToHostMirror
                    (
                        runTime,
                        rho,
                        rhoU,
                        rhoE,
                        U,
                        p,
                        Tgas
                    );
                    resident.downloadNutToHostMirror(runTime, nut);

                    runTime.write();
                    Info<< "runTime = " << runTime.elapsedClockTime()
                        << " simulationTime = " << runTime.timeName()
                        << " particleCount = " << 0
                        << " CoMax = " << lastMeasuredCo << endl;
                }
            }

            return 0;
        }

        GpuParticleResidentSolver resident;
        resident.initialise
        (
            runTime,
            mesh,
            rho,
            rhoU,
            rhoE,
            U,
            p,
            Tgas,
            epsS,
            rhoUs,
            rhoEs,
            rhoDs,
            rhoHp,
            Us,
            theta,
            Tp,
            dMeanCell,
            parcelMass.value(),
            gpuResidentParticleCapacity,
            gpuResidentMaxFaceWalkHops,
            gpuResidentSeed,
            gammaG.value(),
            Rgas.value(),
            strictOpenFoamGasBoundaries,
            rhoS.value(),
            solveParticleTemperature,
            particleThermalRho.value(),
            particleCp.value(),
            muG.value(),
            PrG.value(),
            gasFluxScheme,
            gasReconstruction,
            gasLimiter,
            gasTimeIntegrator,
            gasRobustFallback,
            gasTurbulenceModel,
            gasEntropyFixCoefficient,
            lesDeltaCoeff,
            turbulentPrandtl,
            waleCw,
            smagorinskyCs,
            maxDiffusionNumber,
            dS.value(),
            dMinStrict,
            dMaxStrict,
            dSigmaStrict,
            gpuResidentInjectionTheta,
            gpuResidentInjectionTp,
            rhoMinStrict,
            TgasMinStrict,
            epsSMinStrict,
            thetaMinStrict,
            TpMinStrict,
            TpMaxStrict,
            ugkwpProps
        );
        dispatchScheduledBoundary(resident, runTime.value(), true);

        scalar lastMeasuredCo = scalar(0);
        label stepsSinceCourant = gpuResidentCourantUpdateInterval;

        while (runTime.run())
        {
            #include "readTimeControls.H"

            dispatchScheduledBoundary
            (
                resident,
                runTime.value() + runTime.deltaTValue(),
                false
            );
            const bool courantRefreshDue =
                adjustTimeStep
             && stepsSinceCourant >= gpuResidentCourantUpdateInterval;
            if (courantRefreshDue)
            {
                const scalar oldDt = runTime.deltaTValue();
                lastMeasuredCo =
                    resident.computeGasCourant(oldDt, maxCo);
                if (lastMeasuredCo > SMALL)
                {
                    const scalar requestedDt =
                        oldDt*maxCo/max(lastMeasuredCo, SMALL);
                    const scalar growthLimitedDt =
                        min(requestedDt, oldDt*gpuResidentMaxDeltaTGrowth);
                    runTime.setDeltaT(min(growthLimitedDt, maxDeltaT));
                }
                stepsSinceCourant = 1;
            }
            else
            {
                stepsSinceCourant = min
                (
                    stepsSinceCourant + 1,
                    gpuResidentCourantUpdateInterval
                );
            }

            runTime++;

            resident.advanceOneStep
            (
                runTime.deltaTValue(),
                runTime.value()
            );

            if (runTime.writeTime())
            {
                resident.downloadToHostMirror
                (
                    runTime,
                    rho,
                    rhoU,
                    rhoE,
                    U,
                    p,
                    Tgas,
                    epsS,
                    rhoUs,
                    rhoEs,
                    rhoDs,
                    rhoHp,
                    Us,
                    theta,
                    Tp,
                    dMeanCell
                );
                resident.downloadNutToHostMirror(runTime, nut);

                runTime.write();
                const label currentParticleCount =
                    resident.writeParticleRestartMirror(runTime);
                Info<< "runTime = " << runTime.elapsedClockTime()
                    << " simulationTime = " << runTime.timeName()
                    << " particleCount = " << currentParticleCount
                    << " CoMax = " << lastMeasuredCo << endl;
            }
        }

        return 0;
#endif
    }

#ifdef UGKWP_USE_CUDA
    FatalErrorInFunction
        << "CUDA GPU resident architecture requires gpuResidentStrict=true. "
        << "The CUDA solver build does not allow the legacy CPU active path "
        << "or partial CUDA offload path. CPU/OpenFOAM may only read the case, "
        << "construct the mesh, read dictionaries/initial fields, write host "
        << "mirrors, and restart data."
        << exit(FatalError);
#else
    FatalErrorInFunction
        << "This source tree no longer builds a CPU active UGKWP solver. "
        << "Rebuild with UGKWP_USE_CUDA and set gpuResidentStrict=true. "
        << "CPU/OpenFOAM may only read the case, construct the mesh, read "
        << "dictionaries/initial fields, write host mirrors, and restart data."
        << exit(FatalError);
#endif

    return 1;
}
