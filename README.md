This code forecasts expected numbers of deaths due to COVID-19 in the future,
based on historical numbers of deaths, for a given population (typically a
country or state).  Parameters mentioned below are based on numbers from the
literature; see https://github.com/ketch/covid-blog-posts/wiki/Literature.

The methodology includes the following steps:

1. Based on recorded numbers of deaths for the specified region, infer
   the number of newly infected individuals for each day in the past,
   up to about two weeks ago.  This requires an assumed distribution
   of the time from infection to death, as well as an assumed infection
   fatality ratio (IFR).  Unfortunately, the distribution of times to death
   for deaths on a given day is not the same as the given distribution
   over a whole population, due to varying rates of new infections.
   Instead, formally the rate of infection is related to the rate of death
   by a deconvolution.  This deconvolution is ill-posed and typical
   regularizations give unrealistic predictions, so the model attempts
   to determine just a mean time to death (MTTD) for recent deaths in the
   region and then assumes that each death corresponds to 1/IFR new
   infections that occurred MTTD days earlier.  The assumed IFR is
   0.6% and the MTTD for a uniform distribution of deaths is 17 days.


2. The inferred number of infections goes only up to a day MTTD days in the
   past.  Starting from that day, we simulate forward to the present day using
   the SIR model.  It is necessary to incorporate in this model the effect
   of non-pharmaceutical interventions (NPIs).  Rather than trying to guess
   or estimate the effects of announced measures, since these are likely to
   vary strongly between different regions even for the same measure, the
   model fits the intervention effect to give death counts close to those
   over the corresponding time period.  The intervention is modeled as a
   proportional reduction in the rate of contact between infectious and
   susceptible individuals.  Two fits are done; the first assumes
   a constant level of intervention, while the second fits an intervention
   effect that varies linearly in time.  Because the magnitude of the slope
   is penalized in the fit, these two are fairly close unless there has been
   a recent drastic change in the trend of deaths per day.

3. Using the output of the above model at the present day, and the estimated
   intervention parameter q, the SIR model is again solved forward in time
   to predict future death rates.
