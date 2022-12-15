import numpy as np
UTM_ZONE_LON_WIDTH = 6
MAX_ZONE_LON_WIDTH = 8
MAX_PIXEL_COUNT = 100000

def get_best_utm_zone(longitude: float):
   return np.clip(np.round((longitude + 180) / UTM_ZONE_LON_WIDTH) + 1, 1, 60)
    


def test_get_best_utm_zone():
    assert get_best_utm_zone(15) == 33
    assert get_best_utm_zone(1) == 31
    assert get_best_utm_zone(-179.99) == 1
    assert get_best_utm_zone(169) == 59
    assert get_best_utm_zone(179.99) == 60
