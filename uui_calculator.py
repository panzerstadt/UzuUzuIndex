import sys
sys.path.append('./modules')

print(__file__)
from uzuuzuindex_nn.final_regression_nn import uzu_uzu_index_predictor as uui_nn

"""
latest weather data (ja) = http://www.data.jma.go.jp/obd/stats/data/mdrr/docs/csv_dl_readme.html
historical weather data (ja) = http://www.data.jma.go.jp/gmd/risk/obsdl/index.php
resas api = https://opendata.resas-portal.go.jp/docs/api/v1/index.html

"""

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


for i in changes:
    test_input = ['Sat',5,20.8,0,56,0,1.96,30.6,10,2,0.087407407,0.655]
    uui = uui_nn(new_input=test_input, load_saved_model=True)

    print(uui)