import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch import Tensor


def calculate_max_ray_distances(
    origins: np.ndarray, normalized_directions: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray
) -> np.ndarray:
    """
    Fully vectorized calculation of maximum ray distances using NumPy.

    Args:
        origins: Ray origin points, shape (n_rays, 3)
        normalized_directions: Pre-normalized ray directions, shape (n_rays, 3)
        bounds_min: Minimum bounds of the box, shape (3,)
        bounds_max: Maximum bounds of the box, shape (3,)

    Returns:
        Array of maximum distances for each ray, shape (n_rays,)

    Assumes:
        - All origins are inside the bounds
        - All directions are already normalized
        - All directions are well-defined
    """
    # Calculate both intersection points for each axis simultaneously
    # Shape: (n_rays, 3)
    t1 = (bounds_min - origins) / normalized_directions
    t2 = (bounds_max - origins) / normalized_directions

    # Find min and max intersection points for each axis
    # Shape: (n_rays, 3)
    t_min = np.minimum(t1, t2)
    t_max = np.maximum(t1, t2)

    # Find entry and exit points for each ray
    # Shape: (n_rays,)
    t_enter = np.max(t_min, axis=1)
    t_exit = np.min(t_max, axis=1)

    # Create mask for valid intersections
    # Shape: (n_rays,)
    valid_mask = (t_enter <= t_exit) & (t_exit >= 0)

    # Calculate final distances using vectorized operations
    # For t_enter < 0, use t_exit
    # For t_enter >= 0, use t_exit - t_enter
    result = np.where(valid_mask, np.where(t_enter < 0, t_exit, t_exit - t_enter), 0)

    return result


def calculate_max_ray_distances_pt(
    origins: Tensor, normalized_directions: Tensor, bounds_min: Tensor, bounds_max: Tensor
) -> Tensor:
    """
    Fully vectorized calculation of maximum ray distances using PyTorch.

    Args:
        origins: Ray origin points, shape (n_rays, 3)
        normalized_directions: Pre-normalized ray directions, shape (n_rays, 3)
        bounds_min: Minimum bounds of the box, shape (3,)
        bounds_max: Maximum bounds of the box, shape (3,)

    Returns:
        Array of maximum distances for each ray, shape (n_rays,)

    Assumes:
        - All origins are inside the bounds
        - All directions are already normalized
        - All directions are well-defined
    """
    # Calculate both intersection points for each axis simultaneously
    # Shape: (n_rays, 3)
    t1 = (bounds_min - origins) / normalized_directions
    t2 = (bounds_max - origins) / normalized_directions

    # Find min and max intersection points for each axis
    # Shape: (n_rays, 3)
    t_min = torch.minimum(t1, t2)
    t_max = torch.maximum(t1, t2)

    # Find entry and exit points for each ray
    # Shape: (n_rays,)
    t_enter = torch.max(t_min, dim=-1).values
    t_exit = torch.min(t_max, dim=-1).values

    # Create mask for valid intersections
    # Shape: (n_rays,)
    valid_mask = (t_enter <= t_exit) & (t_exit >= 0)

    # Calculate final distances using vectorized operations
    # For t_enter < 0, use t_exit
    # For t_enter >= 0, use t_exit - t_enter
    result = torch.where(valid_mask, torch.where(t_enter < 0, t_exit, t_exit - t_enter), 0)

    return result


def create_box_visualization(bounds_min, bounds_max):
    """Create vertices and faces for box visualization."""
    vertices = np.array(
        [
            [bounds_min[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_min[1], bounds_min[2]],
            [bounds_max[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_max[1], bounds_min[2]],
            [bounds_min[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_min[1], bounds_max[2]],
            [bounds_max[0], bounds_max[1], bounds_max[2]],
            [bounds_min[0], bounds_max[1], bounds_max[2]],
        ]
    )

    faces = [
        [vertices[j] for j in [0, 1, 2, 3]],
        [vertices[j] for j in [4, 5, 6, 7]],
        [vertices[j] for j in [0, 1, 5, 4]],
        [vertices[j] for j in [2, 3, 7, 6]],
        [vertices[j] for j in [0, 3, 7, 4]],
        [vertices[j] for j in [1, 2, 6, 5]],
    ]

    return faces


def visualize_rays_and_box(
    origins: np.ndarray,
    directions: np.ndarray,
    distances: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    max_rays_to_plot: int = 50,
    alpha: float = 0.3,
):
    """
    Visualize rays, their intersections, and the bounding box.

    Parameters:
    -----------
    origins : np.ndarray
        Array of ray origins
    directions : np.ndarray
        Array of ray directions
    distances : np.ndarray
        Array of maximum distances
    bounds_min, bounds_max : np.ndarray
        Box boundaries
    max_rays_to_plot : int
        Maximum number of rays to plot for clarity
    alpha : float
        Transparency of the box
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot box
    faces = create_box_visualization(bounds_min, bounds_max)
    ax.add_collection3d(Poly3DCollection(faces, alpha=alpha, facecolor="gray", edgecolor="black"))

    # Sample rays if there are too many
    n_rays = len(origins)
    if n_rays > max_rays_to_plot:
        indices = np.random.choice(n_rays, max_rays_to_plot, replace=False)
        origins = origins[indices]
        directions = directions[indices]
        distances = distances[indices]
        n_rays = max_rays_to_plot

    # Plot rays
    for i in range(n_rays):
        if distances[i] > 0:
            end_point = origins[i] + directions[i] * distances[i]
            ax.plot(
                [origins[i, 0], end_point[0]],
                [origins[i, 1], end_point[1]],
                [origins[i, 2], end_point[2]],
                "r-",
                alpha=0.5,
            )
            ax.plot(
                [origins[i, 0]], [origins[i, 1]], [origins[i, 2]], "go", alpha=0.5, label="Origin" if i == 0 else ""
            )
            ax.plot(
                [end_point[0]], [end_point[1]], [end_point[2]], "bo", alpha=0.5, label="Intersection" if i == 0 else ""
            )

    # Set axis labels and limits
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set equal aspect ratio
    ax.set_box_aspect([bounds_max[0] - bounds_min[0], bounds_max[1] - bounds_min[1], bounds_max[2] - bounds_min[2]])

    plt.legend()
    return fig, ax


def benchmark_performance(n_rays_list=[1000, 10000, 100000, 1000000]):
    """
    Benchmark performance with different numbers of rays.
    """
    bounds_min = np.array([-1.0, -1.0, -1.0])
    bounds_max = np.array([1.0, 1.0, 1.0])

    results = []
    for n_rays in n_rays_list:
        # Generate random rays
        origins = np.random.uniform(-2, 2, (n_rays, 3))
        directions = np.random.uniform(-1, 1, (n_rays, 3))

        # Time the calculation
        start_time = time.time()
        _ = calculate_max_ray_distances(origins, directions, bounds_min, bounds_max)
        elapsed_time = time.time() - start_time

        rays_per_second = n_rays / elapsed_time
        results.append({"n_rays": n_rays, "time": elapsed_time, "rays_per_second": rays_per_second})

        print(f"Processed {n_rays:,} rays in {elapsed_time:.4f} seconds " f"({rays_per_second:,.0f} rays/second)")

    return results


# Example usage
def example_with_visualization():
    # Generate sample data
    n_rays = 100
    bounds_min = np.array([-50.0, -50.0, -50.0])
    bounds_max = np.array([50.0, 50.0, 50.0])

    # Generate random origins and directions
    origins = np.random.uniform(-25, 25, (n_rays, 3))  # Start mostly inside box
    directions = np.random.uniform(-1, 1, (n_rays, 3))

    # Calculate distances
    distances = calculate_max_ray_distances(origins, directions, bounds_min, bounds_max)

    # Visualize
    visualize_rays_and_box(origins, directions, distances, bounds_min, bounds_max)
    plt.savefig("test.png")


if __name__ == "__main__":
    example_with_visualization()
    benchmark_performance()
