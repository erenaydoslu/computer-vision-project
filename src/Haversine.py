import torch


def haversine_distance(lat1, lon1, lat2, lon2):
    # Radius of the Earth in kilometers
    R = 6371.0

    # Convert latitude and longitude from degrees to radians
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)

    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
    )
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    distance = R * c
    return distance

def equirectangular_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians using torch
    lat1 = torch.deg2rad(lat1)
    lon1 = torch.deg2rad(lon1)
    lat2 = torch.deg2rad(lat2)
    lon2 = torch.deg2rad(lon2)
    
    # Earth's radius in kilometers
    R = 6371.0
    
    # Differences in coordinates
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    
    # Equirectangular approximation formula using torch
    x = delta_lon * torch.cos((lat1 + lat2) / 2)
    y = delta_lat
    distance = R * torch.sqrt(x * x + y * y)
    
    return distance


class HaversineLoss(torch.nn.Module):
    def __init__(self):
        super(HaversineLoss, self).__init__()

    def forward(self, outputs, targets):
        # Assuming outputs and targets are both of shape (batch_size, 2)
        lat1, lon1 = outputs[:, 0], outputs[:, 1]
        lat2, lon2 = targets[:, 0], targets[:, 1]

        distances = haversine_distance(lat1, lon1, lat2, lon2)
        return distances.mean()

class EquiRectLoss(torch.nn.Module):
    def __init__(self):
        super(EquiRectLoss, self).__init__()

    def forward(self, outputs, targets):
        # Assuming outputs and targets are both of shape (batch_size, 2)
        lat1, lon1 = outputs[:, 0], outputs[:, 1]
        lat2, lon2 = targets[:, 0], targets[:, 1]

        distances = equirectangular_distance(lat1, lon1, lat2, lon2)
        return distances.mean()