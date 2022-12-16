from fuzzywuzzy import fuzz
from unidecode import unidecode

import tensorflow_text
import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3") # TODO: convert to API , remove the need of tf


# Todo: tensorflow
def set_location_id(gmaps_to_std_cities: list, dict_id_city_ctl_locations: dict,
                    dict_match_ratio_locations: dict):  # -> dict, dict
    """
    This function Use ctl_locations.json to standardized locations column by giving its respective id
    Here we use a Tensor Flow encoder to increase our standardization capacity because sometimes Gmaps API
    does not bring the location standardized as we wish it.
    and example of this is that Gmaps returns "Mexico City"and in our ctl_locations.json we have it as "Ciudad de MÉXICO"
    before using tensorflow we try simpler solutions like exact match with  ctl_companies ,fuzzywuzzy
    or already know matches from the past (dict_match_ratio_locations)

    :param gmaps_to_std_cities: cities to standardized
    :param dict_id_city_ctl_locations: dictionary with city to its respective id from ctl_locations
    :param dict_match_ratio_locations: dictionary with already calculated ratios with tensorflow encoder
    """
    # Todo: countries
    try:
        # Todo Remove dict_ratio_companies_file
        dict_location_standarized = {}
        dict_newest_match_ratio_locations = {}

        for gmap_city_ in gmaps_to_std_cities:
            if dict_id_city_ctl_locations.get(gmap_city_) is not None:
                dict_location_standarized[gmap_city_] = dict_id_city_ctl_locations.get(gmap_city_)
                gmaps_to_std_cities = [x for x in gmaps_to_std_cities if x != gmap_city_]

        for gmap_city in gmaps_to_std_cities:
            for city, id in dict_id_city_ctl_locations.items():
                concat = gmap_city + '_' + city
                match_rat_fuz = fuzz.ratio(gmap_city, city)
                # first a fuzzy before tensor flow (faster)
                if match_rat_fuz > 90:
                    dict_location_standarized[gmap_city] = id
                    gmaps_to_std_cities = [x for x in gmaps_to_std_cities if x != gmap_city]
                    break

                # second look if we already did that match in the past
                elif concat not in dict_match_ratio_locations.keys():
                    # if we don't have that match ratio we end up using tensorflow
                    match_rat = int((np.inner(embed(gmap_city), embed(city))[0][0]) * 100)
                    dict_newest_match_ratio_locations[concat] = match_rat
                    if match_rat > 89:
                        dict_location_standarized[gmap_city] = id
                        # we use this break to avoid embeddings of a city that already had a match
                        break
                else:
                    if dict_match_ratio_locations.get(concat) is not None \
                            and dict_match_ratio_locations.get(concat) >= 89:
                        dict_newest_match_ratio_locations[gmap_city] = id
                        # we use this break to avoid embeddings of a city that already had a match in the
                        # dict_embeddings_locations
                        break

    except Exception as error:
        logging.error(f'[ERROR]: Setting Location ID - Error: {error}')

    return dict_location_standarized, dict_newest_match_ratio_locations