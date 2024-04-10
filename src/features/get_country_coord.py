from geopy.geocoders import ArcGIS

def get_country_coord(country_name):
    """
    Retrieves the latitude and longitude coordinates of a given country using geocoding.

    Parameters:
    country_name (str): The name of the country to retrieve the coordinates.

    Returns:
    tuple or None: A tuple containing the latitude and longitude coordinates of the country,
                   or None if the country name is invalid or cannot be geocoded.

    Notes:
    - This function uses geocoding to find the coordinates of the given country name.
    - If the country name is valid and geocoding is successful, the latitude and longitude
      coordinates are returned as a tuple.
    - If the country name is invalid or geocoding fails, the function returns None.

    Example:
    >>> get_country_coord('United States')
    (39.398703156, -99.41461919)
    """
    geolocator = ArcGIS()
    location = geolocator.geocode(country_name)
    if location:
        return location.latitude, location.longitude
    else:
        return None  
      

if __name__ == "__main__":
  import pandas as pd
  df = pd.read_csv('../../data/raw/global_space_launches.csv')
  df.columns
  country = df['Companys Country of Origin']
  unique_country = country.unique()
  only_multi = df[df['Companys Country of Origin']=='Multi']
  
  unique_company_with_multi = only_multi['Company Name'].unique()
  only_multi['Company Name'].value_counts()

  
  country_name = 'Multi'
  coordinates = get_country_coord(country_name)
  print(coordinates)