import numpy as np
import matplotlib.dates as mdates

import covid_forecast as cf
import data
import deconvolution

def test_always_passes():
    assert True

def test_population():
    N = data.get_population('Italy')
    assert(N == 60461.828e3)

    N = data.get_population('Montana')
    assert(N == 1068778)

    N = data.age_distribution['Pakistan'][0]
    assert(N == 27962852.0)

    # Check age breakdown against total
    for region in data.population.keys():
        if region in data.age_distribution.keys():
            N1 = data.population[region]
            N2 = np.sum(data.age_distribution[region])
            assert abs(N1-N2)/N1 < 1e-6
        elif region in ['Andorra','Dominica','Holy See','Liechtenstein','Monaco',
                        'Saint Kitts and Nevis','San Marino']: # Missing from UN demographics
            pass
        else:
            raise Exception('Missing age distribution for '+region)

    for state in data.US_state_population.keys():
        if state == 'Puerto Rico':
            continue
        N1 = np.sum(data.age_distribution_US_states[state])
        N2 = data.get_population(state)
        assert abs(N1-N2)/N1 < 1e-6

def test_ifr():
    assert(abs(cf.avg_ifr('Arizona','mean')-0.009891127911153479)<1.e-7)
    assert(abs(cf.avg_ifr('Arizona','low')-0.005393757717184477)<1.e-7)
    assert(abs(cf.avg_ifr('Arizona','high')-0.01905288977112617)<1.e-7)

def test_SIR():
    u0 = np.array([0.5,0.1,0.4])
    S, I, R = cf.SIR(u0, 0.3, 0.1)
    assert(np.isclose(S[-1],0.2987023148268438))
    assert(np.isclose(I[-1],0.12957483617694002))

def test_loading():
    data_dates, cum_cases, cum_deaths = data.load_time_series('Germany',use_test_data=True)
    assert(cum_cases[-1]==171324)
    assert(cum_deaths[80]==2736)

    data_dates, cum_cases, cum_deaths = data.load_time_series('Washington',use_test_data=True)
    assert(cum_cases[-1]==17537)
    assert(cum_deaths[80]==483)

def test_inference():
    region = 'New York'
    N = data.get_population(region)
    data_dates, cum_cases, cum_deaths = data.load_time_series(region,smooth=True,use_test_data=True)
    data_start = mdates.date2num(data_dates[0])
    ifr = cf.avg_ifr(region)
    gamma = cf.default_gamma
    beta = cf.default_beta
    u0, offset, inferred_data_dates = cf.infer_initial_data(cum_deaths,data_start,ifr,gamma,N)
    assert(offset==20)
    assert(inferred_data_dates[0]==737485.0)
    assert(np.all(np.isclose(u0,np.array([17078249.0462995 ,   770405.00883876,  1604906.94486174]))))
