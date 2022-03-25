import cdsapi

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-land',
    {
        'format': 'netcdf',
        'variable': [
            '10m_v_component_of_wind', '2m_temperature', 'surface_net_solar_radiation',
            'surface_pressure', 'total_precipitation',
        ],
        'year': [
            '2019', '2020', '2021',
        ],
        'month': '07',
        'day': [
            '14', '15', '16',
            '17', '18', '19',
            '20', '21', '22',
            '23', '24',
        ],
        'time': [
            '00:00', '01:00', '02:00',
            '03:00', '04:00', '05:00',
            '06:00', '07:00', '08:00',
            '09:00', '10:00', '11:00',
            '12:00', '13:00', '14:00',
            '15:00', '16:00', '17:00',
            '18:00', '19:00', '20:00',
            '21:00', '22:00', '23:00',
        ],
        'area': [
            22, 72, 16,
            79,
        ],
    },
    'download.nc')