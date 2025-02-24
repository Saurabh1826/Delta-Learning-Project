import xarray as xr

def combine_netcdf_files(file_paths, output_path):
    """
    Combine multiple NetCDF files into a single file.

    Parameters:
    file_paths (list): List of paths to input NetCDF files
    output_path (str): Path where the combined NetCDF file will be saved
    """
    # Load all datasets
    datasets = [xr.open_dataset(file_path) for file_path in file_paths]

    # Merge all datasets
    # The merge operation will align data based on coordinates
    combined_ds = xr.merge(datasets)

    # Save the combined dataset to a new NetCDF file
    combined_ds.to_netcdf(output_path)

    # Close all datasets to free up memory
    for ds in datasets:
        ds.close()

    return combined_ds

# Example usage
file_paths = [
    'data_0.nc',
    'data_1.nc',
    'data_2.nc'
]
output_path = 'combined_data.nc'

# Combine the files
combined_dataset = combine_netcdf_files(file_paths, output_path)
