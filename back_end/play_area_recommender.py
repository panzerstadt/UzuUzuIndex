import sys
sys.path.append('./modules')

print(__file__)
from uzuuzuindex_nn.final_regression_nn import uzu_uzu_index_predictor as uui_nn

"""
latest weather data (ja) = http://www.data.jma.go.jp/obd/stats/data/mdrr/docs/csv_dl_readme.html
historical weather data (ja) = http://www.data.jma.go.jp/gmd/risk/obsdl/index.php
resas api = https://opendata.resas-portal.go.jp/docs/api/v1/index.html

"""

def generate_places():
    places = {}
    places['park_b'] = [0.8,0.7]
    places['park_a'] = [0.6,1.0]
    places['leisure_facility'] = [0.7,0.3]
    places['museum'] = [0.3,0.3]
    places['library'] = [0.2,0.2]
    places['zoo'] = [0.4,0.7]

    return places


def play_recommendation(uui, outdoor_preference, weather_forecast):
    # initial hard coded places
    # todo: train places on the uzu uzu index graph by feedback loop from child
    # e.g. 'did you like it?' 'no' 'why?' 'weather was bad' > detect keyword weather, update preference

    places = generate_places()
    # key[0] = uui
    # kay[1] = outdoor pref
    # larger than uui
    # smaller than outdoor pref
    #print(places)

    # reduce outdoor preference by half in the event of rain (weather < 0.5)
    if weather_forecast > 0.5:
        outdoor_preference = outdoor_preference / 2.0

    recommendations = []
    for key, value in places.items():
        if value[0] >= uui and value[1] <= outdoor_preference:
                recommendations.append(key)

    #print('I recommend: ', recommendations)
    return recommendations


# references
col_labels = ['day of the week',
              'cloud_cover',
              'dew_point',
              'rain_mm',
              'rh',
              'snow_cover',
              'sunlight_hrs',
              'temp_celsius',
              'visibility',
              'weather_rating',
              'forest area coverage',
              'LowPressure']

print('input list must be in the following format: ')
print(col_labels)


# MAIN
# sample child
test_input = ['Sun',5,20.8,0,56,0,1.96,30.6,10,2,0.087407407,0.655]
outdoor_pref = 0.7
forecast_weather = test_input[3]
# currently weather is rain_mm so it is not normalized to 0-1. rain of more than 0.5mm will screw up the recommendation


uui = uui_nn(new_input=test_input, load_saved_model=True)
uui = uui[0]

# output
responses = play_recommendation(uui, outdoor_pref, forecast_weather)

print('recommendations :', responses)
