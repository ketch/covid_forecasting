import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import linalg
import matplotlib.dates as mdates
import data
import json
from scipy import optimize
from utils import NumpyEncoder, smooth
from datetime import datetime, date

"""
Code for modeling and predicting the COVID-19 outbreak.
General nomenclature:
    - cum_* = cumulative values
    - daily_* = daily increments
    - inf = inferred (inference model output)
    - pred = predicted (SIR model output)
"""

default_ifr = 0.006
default_beta=0.27
default_gamma=0.07
hr = 10 # Hospitalizations per death

def ttd_dist():
    "Time to death distribution based on cruise data."
    mean=17; std=7.
    tmin=5; tmax=2*mean+1-tmin;
    nn = tmax+tmin
    assert(2*mean+1==nn)

    t=np.arange(nn)
    p=np.exp(-(t-mean)**2/2/std**2)/np.sqrt(2*np.pi*std**2)
    p[0:tmin]=0.; p[tmax:]=0.
    p=p/np.sum(p)
    return p

def itod_matrix(n):
    p = np.loadtxt('ttd_dist.txt')
    #p = ttd_dist()
    col = np.zeros(n); col[:len(p)]=np.flip(p)
    row = np.zeros(n); row[0]=p[-1]
    return linalg.toeplitz(col,row)


def get_mttd(daily_deaths):
    """
    Determine approximate mean time to death (in days).
    This is not the mean time for an individual, but rather
    the mean time of illness of individuals who died today (for a given region).
    These are quite different when the rate of infection is rapidly increasing
    or decreasing.

    This would more appropriately be done by deconvolution, but I
    haven't found a stable way to do that, so this is an approximate
    way of getting the deconvolved mean.
    """
    mean=17; std=7
    window=mean+std
    t=np.arange(-2*std,2*std+1);
    p=np.exp(-(t)**2/2/std**2)/np.sqrt(2*np.pi*std**2)
    dist = np.convolve(daily_deaths[-window:],p)
    offset = np.sum([(i+0.5)*dist[i] for i in range(len(dist))])/np.sum([dist[i]for i in range(len(dist))])
    return mean-((offset-2*std-window/2))


def SIR(u0, beta=default_beta, gamma=default_gamma, N = 1, T=14, q=0, intervention_start=0, intervention_length=0):
    """
    Run the SIR model with initial data u0 and given parameters.
        - q: intervention strength (1=no human contact; 0=normal contact)
        - intervention_start, intervention_length: measured from simulation start (t=0) in days

    In this version there is only one intervention period.  See SIR2() for a version
    with any number of intervention periods.
    """

    du = np.zeros(3)
    def f(t,u):
        if callable(q):
            qq = q(t)
        else:
            if intervention_start<t<intervention_start+intervention_length:
                qq = q
            else:
                qq = 0.
        du[0] = -(1-qq)*beta*u[1]*u[0]/N
        du[1] = (1-qq)*beta*u[1]*u[0]/N - gamma*u[1]
        du[2] = gamma*u[1]
        return du

    times = np.arange(0,T+1)
    solution = solve_ivp(f,[0,T],u0,t_eval=times,method='RK23',atol=1.e-3,rtol=1.e-3)
    S = solution.y[0,:]
    I = solution.y[1,:]
    R = solution.y[2,:]
    
    return S, I, R


def SIR2(u0, beta=default_beta, gamma=default_gamma, N = 1, T=14, q=[0], intervention_dates=[0,0]):
    """
    Run the SIR model with initial data u0 and given parameters.
        - q: list of intervention strengths (1=no human contact; 0=normal contact)
        - intervention_dates: dates when we switch from one q value to the next.
          First entry is start of first intervention.

    In this version there is only one intervention period.  See SIR2() for a version
    with any number of intervention periods.
    """
    du = np.zeros(3)
    
    def f(t,u):
        i = np.argmax(intervention_dates>t)-1
        if i == -1:
            qq = 0.
        else:
            qq = q[i]

        du[0] = -(1-qq)*beta*u[1]*u[0]/N
        du[1] = (1-qq)*beta*u[1]*u[0]/N - gamma*u[1]
        du[2] = gamma*u[1]
        return du

    times = np.arange(0,T+1)
    # Perhaps we should break this up into one call for each intervention interval.
    solution = solve_ivp(f,[0,T],u0,t_eval=times,method='RK23',atol=1.e-3,rtol=1.e-3)
    S = solution.y[0,:]
    I = solution.y[1,:]
    R = solution.y[2,:]
    
    return S, I, R


def infer_initial_data(cum_deaths,data_start,ifr,gamma,N,extended_output=False):
    """
    Given a sequence of cumulative deaths, infer current values of S, I, and R
    for a population.  The inference dates are offset (backward) from
    the input time series by the mean time to death.

    It is assumed that for each death on day n, there were n/ifr new infections
    on day n-mttd.

    Inputs:
        - cum_deaths: time series of cumulative deaths
        - data_start: starting date of time series
        - ifr: infected fatality ratio
        - gamma: SIR model parameter (1/(time to recovery))
        - N: population size
    Outputs:
        - inferred_data_dates: goes from data_start - mttd up to today
    """
    daily_deaths = np.diff(cum_deaths); daily_deaths = np.insert(daily_deaths,0,cum_deaths[0])
    mttd = int(round(get_mttd(daily_deaths)))

    inferred_data_dates = np.arange(data_start-mttd,data_start+len(cum_deaths))
    cum_deaths = np.insert(cum_deaths,0,[0]*mttd)
    
    inf_daily_infections = np.zeros_like(inferred_data_dates)
    cum_recovered = np.zeros_like(inferred_data_dates)
    
    inf_daily_infections[:-mttd] = daily_deaths/ifr  # Inferred new infections each day

    for i in range(len(inferred_data_dates)):
        cum_recovered[i] = np.sum(inf_daily_infections[:i]*(1-np.exp(-gamma*(i-np.arange(i)))))
    active_infections = np.cumsum(inf_daily_infections) - cum_recovered
    
    
    # Initial values, mttd+1 days ago
    I0 = active_infections[-(mttd+1)]
    R0 = cum_recovered[-(mttd+1)]
    u0 = np.array([N-I0-R0,I0,R0])
    if extended_output:
        return u0, mttd, inferred_data_dates, active_infections, cum_recovered, inf_daily_infections
    else:
        return u0, mttd, inferred_data_dates


def forecast(u0,mttd,N,inferred_data_dates,cum_deaths,ifr=0.007,beta=default_beta,gamma=default_gamma,q=0.,intervention_start=0,
             intervention_length=30,forecast_length=14,death_model='gamma',compute_interval=False):
    """Forecast with SIR model.  All times are in days.

        Inputs:
         - u0: initial data [S, I, R]
         - mttd: difference (in days) between today and simulation start
         - ifr: infection fatality ratio
         - q: measure of NPI
         - intervention_start: when intervention measure starts, relative to today (can be negative)
         - intervention_length (in days from simulation start)
         - if compute_interval is True, then we simulate with a range of parameter values
           and return the min and max values for each day.
        
        Note that the simulation starts from t=0 in simulation time, but that
        is denoted as t=mttd in terms of the inference and prediction dates.

        Also note that the maximum and minimum daily values are not the successive
        differences of the maximum and minimum cumulative values.

        We could change this to just return daily values, since cumulative values
        can be constructed from those.
    """
    S_mean, I_mean, R_mean = SIR(u0, beta=beta, gamma=gamma, N=N, T=mttd+forecast_length, q=q,
                                 intervention_start=intervention_start+mttd,
                                 intervention_length=intervention_length)
    

    prediction_dates = inferred_data_dates[-(mttd+1)]+range(forecast_length+mttd+1)
    if death_model == 'gamma':
        pred_cum_deaths = R_mean*ifr
        # Match values for today:
        pred_cum_deaths = pred_cum_deaths - (pred_cum_deaths[mttd]-cum_deaths[-1])
    elif death_model == 'distribution':
        daily_infections = beta*I_mean*S_mean/N  # Need to include q.
        M = itod_matrix(len(daily_infections));
        pred_daily_deaths = M@daily_infections*ifr
        pred_cum_deaths = np.cumsum(pred_daily_deaths) + cum_deaths[-1]

    if not compute_interval:
        return prediction_dates, pred_cum_deaths, None, None, None, None, S_mean

    else:
        qmin, qmax = compute_interval
        S_low, I_low, R_low = S_mean.copy(), I_mean.copy(), R_mean.copy()
        S_high, I_high, R_high = S_mean.copy(), I_mean.copy(), R_mean.copy()
        dd_low = np.diff(R_mean); dd_high = np.diff(R_mean)

        pred_daily_deaths_low = np.diff(R_mean); pred_daily_deaths_high = np.diff(R_mean)
        for q in np.linspace(qmin,qmax,10):
            S, I, R= SIR(u0, beta=beta, gamma=gamma, N=N, T=mttd+forecast_length, q=q,
                         intervention_start=intervention_start+mttd,
                         intervention_length=intervention_length)

            S_low = np.minimum(S_low,S)
            I_low = np.minimum(I_low,I)
            R_low = np.minimum(R_low,R)
            S_high = np.maximum(S_high,S)
            I_high = np.maximum(I_high,I)
            R_high = np.maximum(R_high,R)
            pred_daily_deaths_low = np.minimum(pred_daily_deaths_low,np.diff(R))
            pred_daily_deaths_high = np.maximum(pred_daily_deaths_high,np.diff(R))
     
        pred_cum_deaths_low  = R_low*ifr
        pred_cum_deaths_low = pred_cum_deaths_low - (pred_cum_deaths_low[mttd]-cum_deaths[-1])
        pred_cum_deaths_high = R_high*ifr
        pred_cum_deaths_high = pred_cum_deaths_high - (pred_cum_deaths_high[mttd]-cum_deaths[-1])

        pred_daily_deaths_low = pred_daily_deaths_low*ifr; pred_daily_deaths_high = pred_daily_deaths_high*ifr

        return prediction_dates, pred_cum_deaths, pred_cum_deaths_low, pred_cum_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high, S_mean


def forecast2(u0,mttd,N,inferred_data_dates,cum_deaths,ifr=0.007,beta=default_beta,gamma=default_gamma,q=[.0],intervention_dates=[0,30],
             forecast_length=14,compute_interval=True):
    """Forecast with SIR model, including multiple intervention periods.  All times are in days.

        Inputs:
         - ifr: infection fatality ratio
         - mttd: difference (in days) between today and simulation start
         - q: measure of NPI
         - intervention_start: when intervention measure starts, relative to today (can be negative)
         - intervention_length (in days from simulation start)
    """

    intervention_dates = np.array(intervention_dates)+mttd
    
    # Now run the model
    S_mean, I_mean, R_mean = SIR2(u0, beta=beta, gamma=gamma, N=N, T=mttd+forecast_length, q=q,intervention_dates=intervention_dates)
    
    prediction_dates = inferred_data_dates[-(mttd+1)]+range(forecast_length+mttd+1)
    pred_cum_deaths = R_mean*ifr
    pred_cum_deaths = pred_cum_deaths - (pred_cum_deaths[mttd]-cum_deaths[-1])

    if not compute_interval:
        return prediction_dates, pred_cum_deaths, None, None, None, None

    else:
        S_low, I_low, R_low = S_mean.copy(), I_mean.copy(), R_mean.copy()
        S_high, I_high, R_high = S_mean.copy(), I_mean.copy(), R_mean.copy()
        dd_low = np.diff(R_mean); dd_high = np.diff(R_mean)

        pred_daily_deaths_low = np.diff(R_mean); pred_daily_deaths_high = np.diff(R_mean)
        for dbeta in np.linspace(-0.05,0.05,20):
            for dgamma in np.linspace(-0.01,0.01,20):
                S, I, R= SIR(u0, beta=beta+dbeta, gamma=gamma+dgamma, N=N, T=mttd+forecast_length, q=q,
                             intervention_start=intervention_start+mttd,
                             intervention_length=intervention_length)

                S_low = np.minimum(S_low,S)
                I_low = np.minimum(I_low,I)
                R_low = np.minimum(R_low,R)
                S_high = np.maximum(S_high,S)
                I_high = np.maximum(I_high,I)
                R_high = np.maximum(R_high,R)
                pred_daily_deaths_low = np.minimum(pred_daily_deaths_low,np.diff(R))
                pred_daily_deaths_high = np.maximum(pred_daily_deaths_high,np.diff(R))
     
        pred_cum_deaths_low  = R_low*ifr
        pred_cum_deaths_low = pred_cum_deaths_low - (pred_cum_deaths_low[mttd]-cum_deaths[-1])
        pred_cum_deaths_high = R_high*ifr
        pred_cum_deaths_high = pred_cum_deaths_high - (pred_cum_deaths_high[mttd]-cum_deaths[-1])

        pred_daily_deaths_low = pred_daily_deaths_low*ifr; pred_daily_deaths_high = pred_daily_deaths_high*ifr

        return prediction_dates, pred_cum_deaths, pred_cum_deaths_low, pred_cum_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high


def plot_forecast(inferred_data_dates, cum_deaths, mttd, prediction_dates, pred_cum_deaths, pred_cum_deaths_low,
                  pred_cum_deaths_high, pred_daily_deaths, pred_daily_deaths_low, pred_daily_deaths_high, plot_title,
                  plot_past_pred=True, plot_type='cumulative',
                  plot_interval=True, plot_value='deaths',scale='linear'):

    if scale == 'linear':
        plotfun = plt.plot_date
    else:
        plotfun = plt.semilogy
        
    if plot_type=='cumulative':
        if plot_value == 'deaths':
            plotfun(inferred_data_dates,cum_deaths,'-',lw=3,label='Deaths (recorded)')
            plotfun(prediction_dates,pred_cum_deaths,'-k',label='Deaths (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates,pred_cum_deaths_low,pred_cum_deaths_high,color='grey',zorder=-1)
        elif plot_value == 'hospitalizations':
            plotfun(prediction_dates,pred_cum_deaths*hr,'-k',label='Hospitalizations (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates,pred_cum_deaths_low*hr,pred_cum_deaths_high*hr,color='grey',zorder=-1)
    elif plot_type=='daily':
        if plot_value == 'deaths':
            plotfun(inferred_data_dates[1:],np.diff(cum_deaths),'-',lw=3,label='Deaths (recorded)')
            plotfun(prediction_dates,pred_daily_deaths,'-k',label='Deaths (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates,pred_daily_deaths_low,pred_daily_deaths_high,color='grey',zorder=-1)
        elif plot_value == 'hospitalizations':
            plotfun(prediction_dates,pred_daily_deaths*hr,'-k',label='Hospitalizations (predicted)')
            if plot_interval:
                plt.fill_between(prediction_dates,pred_daily_deaths_low*hr,pred_daily_deaths_high*hr,color='grey',zorder=-1)

    plt.legend(loc='best')


    ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.title(plot_title)


def compute_and_plot(region='Spain',ifr=0.007,beta=default_beta,gamma=default_gamma,q=0.,
             intervention_start=0,intervention_length=30,forecast_length=14,scale='linear',
             plot_type='cumulative',plot_value='deaths',plot_past_pred=True,plot_interval=True,
             death_model='gamma',estimate_q=False):
    "Shows both past and future."

    N = data.get_population(region)
    data_dates, total_cases, cum_deaths = data.load_time_series(region)
    data_start = mdates.date2num(data_dates[0])  # First day for which we have data

    u0, mttd, inferred_data_dates = infer_initial_data(cum_deaths,data_start,ifr,gamma,N)
    cum_deaths = np.insert(cum_deaths,0,[0]*mttd)

    prediction_dates, pred_daily_deaths, pred_daily_deaths_low, pred_daily_deaths_high, \
        pred_cum_deaths, pred_cum_deaths_low, pred_cum_deaths_high, \
        q_past, immune_fraction, apparent_R, mttd, pred_daily_new_infections = \
        fit_q_and_forecast(region,beta,gamma,ifr,forecast_length,set_q=None)

    plot_title = '{} {}-day forecast with {} for {} days'.format(region,forecast_length,q_past,intervention_length)
    plot_forecast(inferred_data_dates, cum_deaths, mttd, prediction_dates, pred_cum_deaths, pred_cum_deaths_low,
                  pred_cum_deaths_high, pred_daily_deaths, pred_daily_deaths_low, pred_daily_deaths_high,
                  plot_title, plot_past_pred=plot_past_pred, plot_type=plot_type,
                  plot_interval=plot_interval, plot_value=plot_value, scale=scale)



def fit_q_and_forecast(region,beta=0.27,gamma=0.07,ifr=0.007,forecast_length=200,set_q=None,int_len=None):
    """
    Determine an intervention factor (possibly time-dependent) that explains the recent data,
    and then use that to forecast into the future.

    Inputs:
        - set_q:  If 'estimated', use known intervention data.  If 'fitted', fit q to the recent data.
                  If a value or function is given, use that directly.
    """
    # These should be adjusted for each region:
    N = data.get_population(region)

    data_dates, cum_cases, cum_deaths = data.load_time_series(region)
    data_start = mdates.date2num(data_dates[0])  # First day for which we have data

    u0, mttd, inferred_data_dates = infer_initial_data(cum_deaths,data_start,ifr,gamma,N)
    cum_deaths = np.insert(cum_deaths,0,[0]*mttd)
    
    q_past, mttd = assess_intervention_effectiveness(region)
    if set_q == 'estimated':
        q_past = data.estimated_intervention(region)
    if (type(set_q) == float) or callable(set_q):
        q_past = set_q

    if callable(q_past):
        q_current = q_past(mttd)
    else:
        q_current = q_past

    apparent_R = (1-q_current)*beta/gamma
    apparent_R = max(apparent_R,0)

    # Integrate over mttd (in past) to get initial data
    S, I, R= SIR(u0, beta=beta, gamma=gamma, N=N, T=mttd+1, q=q_past, intervention_start=0,
                 intervention_length=mttd*2)

    u0 = np.array([S[-1],I[-1],R[-1]])

    if int_len:
        intervention_length=int_len
    else:
        intervention_length=forecast_length*2
    intervention_start = 0


    q1, mttd = assess_intervention_effectiveness(region,fit_type='linear')
    q2, mttd = assess_intervention_effectiveness(region,fit_type='constant')
    q1 = max(0,min(1,q1(mttd))); q2 = max(0,min(1,q2(mttd)))
    qmin = max(0,min(q1,q2)-0.1)
    qmax = min(1,max(q1,q2)+0.1)

    prediction_dates, pred_cum_deaths, pred_cum_deaths_low, \
      pred_cum_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high, \
      S = forecast(u0,0,N,[inferred_data_dates[-1]],cum_deaths,ifr,beta,gamma,
                 q_current,intervention_start,intervention_length,forecast_length,'gamma',compute_interval=[qmin,qmax])
    
    pred_daily_new_infections = - np.diff(S)
    pred_daily_deaths = np.diff(pred_cum_deaths);
    immune_fraction = (N-S)/N

    return prediction_dates[1:], pred_daily_deaths, pred_daily_deaths_low, pred_daily_deaths_high, \
            pred_cum_deaths[1:], pred_cum_deaths_low[1:], pred_cum_deaths_high[1:], \
            q_current, immune_fraction, apparent_R, mttd, pred_daily_new_infections


def get_past_infections(region,ifr=default_ifr,beta=default_beta,gamma=default_gamma):
    N = data.get_population(region)
    data_dates, cum_cases, cum_deaths = data.load_time_series(region)
    data_start = mdates.date2num(data_dates[0])  # First day for which we have data

    u0, mttd, inf_dates, inf_I, inf_R, inf_newI = \
        infer_initial_data(cum_deaths,data_start,ifr,gamma,N,extended_output=1)

    q_past, mttd = assess_intervention_effectiveness(region)
    S, I, R= SIR(u0, beta, gamma, N=N, T=mttd, q=q_past, intervention_start=0,
                 intervention_length=mttd*2)
    inf_newI[-mttd:] = -np.diff(S)
    return inf_dates, inf_newI


def write_JSON(regions, forecast_length=200, print_estimates=False):

    output = {}
    ifr = default_ifr
    gamma = default_gamma
    beta = default_beta

    for region in regions:

        data_dates, cum_cases, cum_deaths = data.load_time_series(region)
        if cum_deaths[-1]<50: continue

        prediction_dates, pred_daily_deaths, pred_daily_deaths_low, pred_daily_deaths_high, \
            pred_cum_deaths, pred_cum_deaths_low, pred_cum_deaths_high, \
            q_past, immune_fraction, apparent_R, mttd, pred_daily_new_infections = \
            fit_q_and_forecast(region,beta,gamma,ifr,forecast_length)

        #q_est = data.estimated_intervention(region)
        q1, mttd = assess_intervention_effectiveness(region,fit_type='linear')
        q2, mttd = assess_intervention_effectiveness(region,fit_type='constant')
        q1 = max(0,min(1,q1(mttd))); q2 = max(0,min(1,q2(mttd)))

        N = data.get_population(region)
        estimated_immunity = immune_fraction[0]
        if print_estimates:
            print('{:>15}: {:.2f} {:.2f} {:.3f}'.format(region,q1,q2, estimated_immunity))

        past_dates, past_new_infections = get_past_infections(region,ifr,beta,gamma)

        prediction_dates = [datetime.strftime(mdates.num2date(ddd),"%m/%d/%Y") for ddd in prediction_dates]
        past_dates = [datetime.strftime(mdates.num2date(ddd),"%m/%d/%Y") for ddd in past_dates]

        output[region] = {}
        output[region]['dates'] = prediction_dates
        output[region]['new infections'] = pred_daily_new_infections
        output[region]['deaths'] = pred_daily_deaths
        output[region]['deaths_low'] = pred_daily_deaths_low
        output[region]['deaths_high'] = pred_daily_deaths_high
        output[region]['intervention effectiveness'] = q_past
        output[region]['intervention effectiveness interval'] = mttd
        output[region]['estimated immunity'] = estimated_immunity
        output[region]['past dates'] = past_dates
        output[region]['past new infections'] = past_new_infections
        
    with open('./output/forecast_{}.json'.format(date.today()), 'w') as file:
        json.dump(output, file, cls=NumpyEncoder)


def assess_intervention_effectiveness(region, plot_result=False, slope_penalty=65, fit_type='linear'):
    """
    For a given region, determine an intervention effectiveness q(t) that
    reproduces the recent data as accurately as possible.
    Inputs:
        - fit_type: 'constant' or 'linear'
        - slope_penalty: multiplies penalty term for linear fit; has no effect for constant fit.
            In principle, taking a large slope penalty should be equivalent to doing a constant
            fit, but do to failure of the optimizer this is not true in practice.
        - plot_result: if True, show the data and the fit.
    """
    ifr = default_ifr

    beta = default_beta
    gamma = default_gamma

    N = data.get_population(region)
    data_dates, total_cases, cum_deaths = data.load_time_series(region)
    data_start = mdates.date2num(data_dates[0])  # First day for which we have data

    u0, mttd, inferred_data_dates = infer_initial_data(cum_deaths,data_start,ifr,gamma,N)
    cum_deaths = np.insert(cum_deaths,0,[0]*mttd)

    intervention_start=-mttd # Could just set -infinity
    intervention_length=mttd
    forecast_length=0

    def fit_q_linear(v):
        "Fit a linearly-varying (in time) intervention factor to the recent data."
        q0 = v[0]
        slope = v[1]
        penalty = 100
        qfun = lambda t: q0+slope*t
        prediction_dates, pred_cum_deaths, pred_cum_deaths_low, \
          pred_cum_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high, S \
          = forecast(u0,mttd,N,inferred_data_dates,cum_deaths,ifr,beta,gamma,qfun,
                     intervention_start,intervention_length,forecast_length,'gamma',False)
        log_daily_deaths = np.log(np.maximum(np.diff(cum_deaths)[-mttd:],1.e0))
        endval = qfun(mttd)
        residual = np.linalg.norm(np.arange(mttd)*(np.log(np.diff(pred_cum_deaths)) \
                    - smooth(log_daily_deaths))) + slope_penalty*abs(v[1]) + penalty*(max(0,endval-1) - min(0,endval))
        return residual

    def fit_q_constant(q):
        "Fit a constant (in time) intervention factor to the recent data."
        prediction_dates, pred_cum_deaths, pred_cum_deaths_low, \
          pred_cum_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high, S \
          = forecast(u0,mttd,N,inferred_data_dates,cum_deaths,ifr,beta,gamma,q,
                     intervention_start,intervention_length,forecast_length,'gamma',False)

        log_daily_deaths = np.log(np.maximum(np.diff(cum_deaths)[-mttd:],1.e-0))
        residual = np.linalg.norm(np.arange(mttd)**2*(np.log(np.diff(pred_cum_deaths))-smooth(log_daily_deaths)))
        return residual

    if fit_type == 'linear':
        bounds = ((0.,1.),(-0.1,0.1))
        result = optimize.minimize(fit_q_linear,[0.,0.],bounds=bounds,method='slsqp')
        qfun = lambda t: result.x[0] + t*result.x[1]
    elif fit_type == 'constant':
        q = optimize.fsolve(fit_q_constant,0.,epsfcn=0.01)[0]
        qfun = lambda t: q

    if plot_result:
        prediction_dates, pred_cum_deaths, pred_cum_deaths_low, \
          pred_cum_deaths_high, pred_daily_deaths_low, pred_daily_deaths_high, S \
          = forecast(u0,mttd,N,inferred_data_dates,cum_deaths,ifr,beta,gamma,qfun,
                     intervention_start,intervention_length,forecast_length,'gamma',False)

        plt.semilogy(prediction_dates[1:],np.diff(pred_cum_deaths))
        plt.semilogy(prediction_dates[1:],np.diff(cum_deaths[-mttd-1:]))

    return qfun, mttd

if __name__ == '__main__':

    import warnings
    warnings.filterwarnings('ignore')
    regions = list(data.population.keys()) + list(data.US_state_population.keys())
    write_JSON(regions,print_estimates=True)
