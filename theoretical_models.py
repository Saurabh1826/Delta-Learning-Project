import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

## Base Model
def calculate_soil_temp_base_model(current_air_temp, previous_air_temp, has_snow_cover, regional_soil_temp):
    """
    Calculate the modified soil temperature for a given day based on air temperatures,
    snow cover, and regional soil temperature estimates.

    Parameters:
    -----------
    current_air_temp : float
        The current day's air temperature A(J)
    previous_air_temp : float
        The previous day's air temperature A(J-1)
    has_snow_cover : bool
        True if there is snow cover, False if there is no snow
    regional_soil_temp : float
        The soil temperature estimated from regional equations E(J)

    Returns:
    --------
    float
        The modified soil temperature F(J) for the current day
    """
    #rate scaler (M1) based on snow cover
    rate_scaler = 0.1 if has_snow_cover else 0.25

    # temp difference term
    temp_difference = current_air_temp - previous_air_temp

    # F(J) = [A(J) - A(J-1)] * M1 + E(J)
    modified_soil_temp = (temp_difference * rate_scaler) + regional_soil_temp

    return modified_soil_temp

## Vegetation Model
def calculate_soil_temp_under_vegetation(previous_soil_temp, current_air_temp, lai, k=0.5, m2=0.25):
    """
    Calculate the soil temperature under vegetation using the formula:
    T(J) = T(J-1) + [A(J) - T(J-1)] * M2 * exp(-K * LAI)

    This formula accounts for how vegetation cover affects soil temperature through:
    1. Shading (represented by LAI - Leaf Area Index)
    2. Light extinction (K coefficient)
    3. Temperature difference between air and previous soil temperature

    Parameters:
    -----------
    previous_soil_temp : float
        The soil temperature from the previous day T(J-1)
    current_air_temp : float
        The current day's air temperature A(J)
    lai : float
        Leaf Area Index - represents density of leaf coverage
    k : float, optional
        Extinction coefficient (defaults to 0.5 for forests)
    m2 : float, optional
        Rate scaler (defaults to 0.25)

    Returns:
    --------
    float
        The calculated soil temperature under vegetation T(J)
    """
    # temp difference between air and previous soil temp
    temp_difference = current_air_temp - previous_soil_temp

    # calc the vegetation effect using extinction coefficient and LAI
    vegetation_effect = np.exp(-k * lai)

    # calc the rate of temperature change
    # This combines the constant rate scaler (M2) with the vegetation effect
    rate_of_change = m2 * vegetation_effect

    # T(J) = T(J-1) + [A(J) - T(J-1)] * [M2 * exp(-K * LAI)]
    new_soil_temp = previous_soil_temp + (temp_difference * rate_of_change)

    return new_soil_temp

class SoilTempDataProcessor:
    def __init__(self, netcdf_path):
        """
        Initialize the processor with a NetCDF file path.
        Handles all available data variables: air temp, soil temp, snow cover,
        leaf index, soil water, and precipitation
        """
        self.ds = xr.open_dataset(netcdf_path)
        self.df = self.ds.to_dataframe().reset_index()

        # Map expected variable names to actual names in your NetCDF file
        # Update these mappings based on your actual variable names
        self.var_mapping = {
            'air_temp': 't2m',         # 2m air temperature
            'soil_temp': 'stl4',       # soil temperature
            'snow_cover': 'snowc',      # snow cover (update to actual variable name)
            'leaf_index': 'lai_lv',       # leaf area index (update to actual variable name)
            # 'soil_water': 'sw',        # soil water (update to actual variable name)
            # 'precipitation': 'precip'   # precipitation (update to actual variable name)
        }
    def _get_var_name(self, var_key):
        """Get the actual variable name from the mapping."""
        return self.var_mapping.get(var_key, var_key)

    def prepare_time_series_data(self, location):
        """
        Prepare time series data for a specific lat/lon location.

        Parameters:
        -----------
        location : tuple
            (latitude, longitude) tuple for the point of interest

        Returns:
        --------
        dict
            Dictionary containing processed data arrays for all variables
        """
        lat, lon = location

        # Find closest point to requested location
        df_location = self.df[
            (self.df['latitude'] == self.ds.latitude.sel(latitude=lat, method='nearest').item()) &
            (self.df['longitude'] == self.ds.longitude.sel(longitude=lon, method='nearest').item())
        ]

        # Sort by time
        df_location = df_location.sort_values('valid_time')

        # Process all variables
        processed_data = {
            'times': df_location['valid_time'].values[1:],  # Exclude first timestamp
        }

        # Add current and previous values for each variable
        for var_key in self.var_mapping:
            var_name = self._get_var_name(var_key)
            if var_name in df_location.columns:
                values = df_location[var_name].values
                processed_data[f'current_{var_key}'] = values[1:]    # All except first
                processed_data[f'previous_{var_key}'] = values[:-1]  # All except last

        return processed_data

    def get_base_model_inputs(self, location):
        """
        Get data formatted specifically for the base model.
        Now includes actual snow cover data.
        """
        data = self.prepare_time_series_data(location)
        return {
            'current_air_temp': data['current_air_temp'],
            'previous_air_temp': data['previous_air_temp'],
            'has_snow_cover': data['current_snow_cover'] > 0,  # Convert to boolean based on threshold
            'regional_soil_temp': data['current_soil_temp']
        }

    def get_vegetation_model_inputs(self, location, k=0.5, m2=0.25):
        """
        Get data formatted specifically for the vegetation model.
        Includes LAI data and optional parameters.
        """
        data = self.prepare_time_series_data(location)
        return {
            'previous_soil_temp': data['previous_soil_temp'],
            'current_air_temp': data['current_air_temp'],
            'lai': data['current_leaf_index'],
            'k': np.full_like(data['current_air_temp'], k),
            'm2': np.full_like(data['current_air_temp'], m2)
        }

    def run_base_model(self, location, model_func):
        """
        Run the base model for a specific location.

        Returns:
        --------
        tuple
            (times, temperatures) - Array of timestamps and corresponding modeled temperatures
        """
        inputs = self.get_base_model_inputs(location)
        results = []

        for i in range(len(inputs['current_air_temp'])):
            result = model_func(
                inputs['current_air_temp'][i],
                inputs['previous_air_temp'][i],
                inputs['has_snow_cover'][i],
                inputs['regional_soil_temp'][i]
            )
            results.append(result)

        return self.prepare_time_series_data(location)['times'], np.array(results)

    def run_vegetation_model(self, location, model_func, k=0.5, m2=0.25):
        """
        Run the vegetation model for a specific location.

        Returns:
        --------
        tuple
            (times, temperatures) - Array of timestamps and corresponding modeled temperatures
        """
        inputs = self.get_vegetation_model_inputs(location, k, m2)
        results = []

        for i in range(len(inputs['current_air_temp'])):
            result = model_func(
                inputs['previous_soil_temp'][i],
                inputs['current_air_temp'][i],
                inputs['lai'][i],
                inputs['k'][i],
                inputs['m2'][i]
            )
            results.append(result)

        return self.prepare_time_series_data(location)['times'], np.array(results)
    
    def compare_models(self, location, base_model_func, vegetation_model_func, k=0.5, m2=0.25):
        """
        Run and compare both models for the same location.

        Returns:
        --------
        dict
            Dictionary containing times and results from both models
        """
        times, base_results = self.run_base_model(location, base_model_func)
        _, veg_results = self.run_vegetation_model(location, vegetation_model_func, k, m2)

        actual_temps = self.prepare_time_series_data(location)['current_soil_temp']
        return {
            'times': times,
            'base_model': base_results,
            'vegetation_model': veg_results,
            'actual': actual_temps,
            'base_model_rmse': np.sqrt(np.mean((base_results - actual_temps) ** 2)),
            'vegetation_model_rmse': np.sqrt(np.mean((veg_results - actual_temps) ** 2))
        }

if __name__=="__main__":

    processor = SoilTempDataProcessor('combined_data.nc')
    
    # # Define a location of interest (latitude, longitude)
    # location = (45.0, -90.0)

    # # Run both models and compare results
    # results = processor.compare_models(
    #     location=location,
    #     base_model_func=calculate_soil_temp_base_model,
    #     vegetation_model_func=calculate_soil_temp_under_vegetation
    # )

    # embed()
    
    # # Plot the results
    # plt.figure(figsize=(12, 6))
    # plt.plot(results['times'], results['actual'], label='Actual', color='black')
    # plt.plot(results['times'], results['base_model'], label=f'Base Model (RMSE: {results["base_model_rmse"]:.2f})')
    # plt.plot(results['times'], results['vegetation_model'], label=f'Vegetation Model (RMSE: {results["vegetation_model_rmse"]:.2f})')
    # plt.xlabel('Time')
    # plt.ylabel('Temperature')
    # plt.legend()
    # plt.title('Comparison of Soil Temperature Models')

    # plt.savefig('theory.png')


    xrds=xr.open_dataset("combined_data.nc")
    df1=xrds.to_dataframe()

    stl_data = xrds['stl1'].to_numpy()
    t2m_data = xrds['t2m'].to_numpy()
    latitude = xrds['latitude'].to_numpy()
    longitude = xrds['longitude'].to_numpy()

    
    ### Calculate theoretical temperatures
    print('calculating')
    processor = SoilTempDataProcessor('combined_data.nc')
    base_model = []
    vegetation_model = []
    for i, lat in enumerate(latitude):
        row_b, row_v = [], []
        for lon in longitude:
            theory_results = processor.compare_models(location=(lat,lon),
                                                      base_model_func=calculate_soil_temp_base_model,
                                                      vegetation_model_func=calculate_soil_temp_under_vegetation)
            row_b.append(theory_results['base_model'])
            row_v.append(theory_results['vegetation_model'])
        base_model.append(row_b)
        vegetation_model.append(row_v)
        print(i)
    base_model = np.array(base_model)
    vegetation_model = np.array(vegetation_model)
