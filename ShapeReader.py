"""This script reads nuclear shape parameters from a file and generates plots."""

import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from scipy.special import sph_harm
from dataclasses import dataclass
from typing import Tuple

matplotlib.use('TkAgg')

# Constants
r0 = 1.16  # Radius constant in fm


@dataclass
class NuclearEvent:
    """Class to store nuclear event parameters."""
    event_category: str
    number_of_protons: int
    number_of_neutrons: int
    mass: float
    total_energy: float
    liquid_drop_energy: float
    shell_correction_energy: float
    rotational_energy: float
    beta_parameters: Tuple[float, ...]  # β10-β60, β70-β80 will be added as zeros
    fission_theta: float
    fission_ratio: float
    step_count: int

    def get_full_beta_parameters(self) -> Tuple[float, ...]:
        """Return all beta parameters including zeros for β70 and β80."""
        return self.beta_parameters + (0.0, 0.0)


def calculate_volume(number_of_protons, number_of_neutrons, parameters):
    """
    Calculate the volume of a nucleus with given parameters.

    Args:
    :parameter    Z (int): The number of protons.
    :parameter    N (int): The number of neutrons.
    :parameter    parameters (tuple): A tuple of deformation parameters (beta10, beta20, ..., beta80).

    Returns:
    :return    float: The calculated volume of the nucleus.
    """
    number_of_nucleons = number_of_protons + number_of_neutrons
    beta10, beta20, beta30, beta40, beta50, beta60, beta70, beta80 = parameters

    # Base coefficient
    base_coefficient = 1 / (111546435 * np.sqrt(np.pi))

    # Main terms
    term1 = 148728580 * np.pi ** (3 / 2)
    term2 = 22309287 * np.sqrt(5) * beta10 ** 2 * beta20
    term3 = 5311735 * np.sqrt(5) * beta20 ** 3
    term4 = 47805615 * beta20 ** 2 * beta40
    term5 = 30421755 * beta30 ** 2 * beta40
    term6 = 9026235 * beta40 ** 3
    term7 = 6686100 * np.sqrt(77) * beta30 * beta40 * beta50
    term8 = 25741485 * beta40 * beta50 ** 2
    term9 = 13000750 * np.sqrt(13) * beta30 ** 2 * beta60
    term10 = 7800450 * np.sqrt(13) * beta40 ** 2 * beta60

    # Additional terms
    term11 = 1820105 * np.sqrt(1001) * beta30 * beta50 * beta60
    term12 = 6729800 * np.sqrt(13) * beta50 ** 2 * beta60
    term13 = 25053210 * beta40 * beta60 ** 2
    term14 = 2093000 * np.sqrt(13) * beta60 ** 3
    term15 = 9100525 * np.sqrt(105) * beta30 * beta40 * beta70

    # More complex terms
    term16 = 4282600 * np.sqrt(165) * beta40 * beta50 * beta70
    term17 = 1541736 * np.sqrt(1365) * beta30 * beta60 * beta70
    term18 = 1014300 * np.sqrt(2145) * beta50 * beta60 * beta70
    term19 = 24647490 * beta40 * beta70 ** 2
    term20 = 6037500 * np.sqrt(13) * beta60 * beta70 ** 2

    # Beta80 terms
    term21 = 11241825 * np.sqrt(17) * beta40 ** 2 * beta80
    term22 = 2569560 * np.sqrt(1309) * beta30 * beta50 * beta80
    term23 = 6508425 * np.sqrt(17) * beta50 ** 2 * beta80
    term24 = 3651480 * np.sqrt(221) * beta40 * beta60 * beta80
    term25 = 5494125 * np.sqrt(17) * beta60 ** 2 * beta80

    # Final terms
    term26 = 1338876 * np.sqrt(1785) * beta30 * beta70 * beta80
    term27 = 869400 * np.sqrt(2805) * beta50 * beta70 * beta80
    term28 = 5053125 * np.sqrt(17) * beta70 ** 2 * beta80
    term29 = 24386670 * beta40 * beta80 ** 2
    term30 = 5890500 * np.sqrt(13) * beta60 * beta80 ** 2
    term31 = 1603525 * np.sqrt(17) * beta80 ** 3

    # Sum of squares term
    squares_sum = 111546435 * np.sqrt(np.pi) * (
            beta10 ** 2 + beta20 ** 2 + beta30 ** 2 + beta40 ** 2 +
            beta50 ** 2 + beta60 ** 2 + beta70 ** 2 + beta80 ** 2
    )

    # Beta10 related terms
    beta10_term = 437 * beta10 * (
            21879 * np.sqrt(105) * beta20 * beta30 +
            48620 * np.sqrt(21) * beta30 * beta40 +
            7 * (
                    5525 * np.sqrt(33) * beta40 * beta50 +
                    1530 * np.sqrt(429) * beta50 * beta60 +
                    3927 * np.sqrt(65) * beta60 * beta70 +
                    3432 * np.sqrt(85) * beta70 * beta80
            )
    )

    # Beta20 related terms
    beta20_term = 23 * beta20 * (
            646646 * np.sqrt(5) * beta30 ** 2 +
            629850 * np.sqrt(5) * beta40 ** 2 +
            209950 * np.sqrt(385) * beta30 * beta50 +
            621775 * np.sqrt(5) * beta50 ** 2 +
            508725 * np.sqrt(65) * beta40 * beta60 +
            712215 * np.sqrt(33) * beta50 * beta70 +
            21 * np.sqrt(5) * (
                    29393 * beta60 ** 2 +
                    29260 * beta70 ** 2 +
                    5852 * np.sqrt(221) * beta60 * beta80 +
                    29172 * beta80 ** 2
            )
    )

    # Sum all terms
    total = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 +
             term11 + term12 + term13 + term14 + term15 + term16 + term17 + term18 + term19 + term20 +
             term21 + term22 + term23 + term24 + term25 + term26 + term27 + term28 + term29 + term30 +
             term31 + squares_sum + beta10_term + beta20_term)

    # Final calculation
    volume = base_coefficient * number_of_nucleons * r0 ** 3 * total

    # print(volume)

    return volume


def calculate_sphere_volume(number_of_protons, number_of_neutrons):
    """
    Calculate the volume of a spherical nucleus.

    Args:
    :parameter    Z (int): The number of protons.
    :parameter    N (int): The number of neutrons.

    Returns:
    :return    float: The calculated volume of the spherical nucleus.
    """
    sphere_volume = 4 / 3 * np.pi * (number_of_protons + number_of_neutrons) * r0 ** 3

    # print(sphere_volume)

    return sphere_volume


def calculate_volume_fixing_factor(number_of_protons, number_of_neutrons, parameters):
    """
    Calculate the volume fixing factor to conserve volume.

    Args:
    :parameter    Z (int): The number of protons.
    :parameter    N (int): The number of neutrons.
    :parameter    parameters (tuple): A tuple of deformation parameters.

    Returns:
    :return    float: The volume fixing factor.
    """
    # Calculate the volume of the initial shape
    initial_volume = calculate_volume(number_of_protons, number_of_neutrons, parameters)

    # Calculate the volume of the sphere
    sphere_volume = calculate_sphere_volume(number_of_protons, number_of_neutrons)

    # Calculate the volume fixing factor
    volume_fix = (sphere_volume / initial_volume)

    # print(volume_fix)

    return volume_fix


def calculate_radius(theta, parameters, number_of_protons, number_of_neutrons):
    """
    Calculate the nuclear radius as a function of polar angle theta.

    Args:
    :parameter    theta (np.ndarray): An array of polar angles.
    :parameter    parameters (tuple): number_of_nucleons tuple of deformation parameters.
    :parameter    Z (int): The number of protons.
    :parameter    N (int): The number of neutrons.

    Returns:
    :return    np.ndarray: The calculated nuclear radius for each theta.
    """
    # Base shape from spherical harmonics
    radius = np.ones_like(theta)

    # print(parameters)

    for harmonic_index in range(1, 9):
        # Using only the m=0 harmonics (axially symmetric)
        harmonic = np.real(sph_harm(0, harmonic_index, 0, theta))
        radius += parameters[harmonic_index - 1] * harmonic

    # Calculate radius correction factor
    radius_fix = calculate_volume_fixing_factor(number_of_protons, number_of_neutrons, parameters) ** (1 / 3)

    # Apply number_of_nucleons^(1/3) scaling and volume conservation
    number_of_nucleons = number_of_protons + number_of_neutrons
    nuclear_radius = 1.16 * (number_of_nucleons ** (1 / 3)) * radius_fix * radius

    # Check if the calculated radius is not negative
    # if np.any(nuclear_radius < 0):
    #    print("Negative radius detected!")

    return nuclear_radius


def calculate_volume_by_integration(number_of_protons, number_of_neutrons, parameters):
    """
    Calculate the volume of the nucleus by numerical integration.

    Args:
    :parameter Z (int): Number of protons
    :parameter N (int): Number of neutrons
    :parameter parameters (tuple): Deformation parameters (beta10, beta20, ..., beta80)

    Returns:
    :return float: Volume in fm³
    """
    # Number of points for integration
    n_theta = 200
    n_phi = 200

    # Integration variables
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    theta_mesh, phi_mesh = np.meshgrid(theta, phi)

    # Calculate r(theta) for all theta values
    r = calculate_radius(theta_mesh, parameters, number_of_protons, number_of_neutrons)

    # Volume element in spherical coordinates: r²sin(θ)drdθdφ
    # Since we're integrating from 0 to r(θ,φ), the r integral gives us r³/3
    integrand = (r ** 3 * np.sin(theta_mesh)) / 3

    # Numerical integration using trapezoidal rule
    volume = np.trapezoid(np.trapezoid(integrand, theta, axis=1), phi)

    # print(volume)

    return volume


def find_neck_thickness(x_coords, y_coords, theta_vals, degree_range):
    """
    Find the neck thickness - shortest distance from x-axis between specified degree range.

    Args:
    :parameter x_coords (np.ndarray): x coordinates of the nuclear shape
    :parameter y_coords (np.ndarray): y coordinates of the nuclear shape
    :parameter theta_vals (np.ndarray): theta values used for plotting
    :parameter degree_range (tuple): (start_degree, end_degree) for neck calculation

    Returns:
    :return tuple: (neck_thickness, neck_x, neck_y) - the neck thickness and its coordinates
    """
    # Convert degree range to radians
    start_rad, end_rad = np.radians(degree_range)

    # Find indices corresponding to theta within the specified degree range
    mask = (theta_vals >= start_rad) & (theta_vals <= end_rad)
    relevant_x = x_coords[mask]
    relevant_y = y_coords[mask]

    # Calculate distances from x-axis (absolute y values)
    distances = np.abs(relevant_y)

    # Find the minimum distance and its index
    neck_idx = np.argmin(distances)
    neck_thickness = distances[neck_idx] * 2  # Multiply by 2 for full thickness
    neck_x = relevant_x[neck_idx]
    neck_y = relevant_y[neck_idx]

    return neck_thickness, neck_x, neck_y


def parse_line(line: str) -> NuclearEvent:
    """
    Parse a line from the input file into a NuclearEvent object.

    Args:
    :parameter line (str): A line from the input file

    Returns:
    :return NuclearEvent: Object containing all event parameters
    """
    values = line.strip().split()
    if len(values) != 17:  # Updated for 6 beta parameters
        raise ValueError(f"Expected 17 values, got {len(values)}")

    # print(values)

    return NuclearEvent(
        event_category=values[0],
        number_of_protons=int(values[1]),
        number_of_neutrons=int(values[2]),
        mass=float(values[3]),
        total_energy=float(values[4]),
        liquid_drop_energy=float(values[5]),
        shell_correction_energy=float(values[6]),
        rotational_energy=float(values[7]),
        beta_parameters=tuple(float(x) for x in values[8:14]),  # β10-β60
        fission_theta=float(values[14]),
        fission_ratio=float(values[15]),
        step_count=int(values[16])
    )


def find_nearest_point(plot_x, plot_y, angle):
    """
    Find the nearest point on the curve to a given angle.

    Args:
    :parameter plot_x (np.ndarray): The x-coordinates of the plot
    :parameter plot_y (np.ndarray): The y-coordinates of the plot
    :parameter angle (float): The target angle in radians

    Returns:
    :return tuple: The x and y coordinates of the nearest point
    """
    angles = np.arctan2(plot_y, plot_x)
    angle_diff = np.abs(angles - angle)
    nearest_index = np.argmin(angle_diff)
    return plot_x[nearest_index], plot_y[nearest_index]


def create_plot(event: NuclearEvent, output_filename: str, event_number: int, input_filename: str):
    """
    Create and save a plot for the given nuclear event.

    Args:
    event (NuclearEvent): Nuclear event parameters
    outputFilename (str): Name of the output file
    eventNumber (int): Number of the event (line number)
    inputFilename (str): Name of the input file for extracting parameters
    """
    # Create figure with two subplots
    fig = plt.figure(figsize=(15, 8))
    ax_plot = fig.add_subplot(121)
    ax_text = fig.add_subplot(122)
    ax_text.axis('off')

    # Parse input filename for additional parameters
    file_params = parse_filename(input_filename)

    # Get complete beta parameters including zeros for β70 and β80
    full_beta_parameters = event.get_full_beta_parameters()

    # Generate theta values and calculate radius
    theta = np.linspace(0, 2 * np.pi, 2000)
    plot_radius = calculate_radius(theta, full_beta_parameters,
                                   event.number_of_protons, event.number_of_neutrons)
    plot_x = plot_radius * np.cos(theta)
    plot_y = plot_radius * np.sin(theta)

    # Plot the shape
    ax_plot.plot(plot_x, plot_y)

    # Find intersection points with axes
    x_axis_positive = find_nearest_point(plot_x, plot_y, 0)
    x_axis_negative = find_nearest_point(plot_x, plot_y, np.pi)
    y_axis_positive = find_nearest_point(plot_x, plot_y, np.pi / 2)
    y_axis_negative = find_nearest_point(plot_x, plot_y, -np.pi / 2)

    # Draw axis lines and set labels
    ax_plot.plot([x_axis_negative[0], x_axis_positive[0]],
                 [x_axis_negative[1], x_axis_positive[1]],
                 color='red', label='X-axis')
    ax_plot.plot([y_axis_negative[0], y_axis_positive[0]],
                 [y_axis_negative[1], y_axis_positive[1]],
                 color='blue', label='Y-axis')

    # Calculate and add text for nuclear parameters
    nuclear_params = (
        f'Z={file_params["number_of_protons"]}, N={file_params["number_of_neutrons"]}\n'
        f'E*={file_params["excitation_energy"]} MeV, J={file_params["angular_momentum"]}ℏ\n'
        f'Event Category: {event.event_category}'
    )

    # Add title with nuclear parameters
    ax_plot.set_title(nuclear_params, fontsize=16, pad=20)

    # Calculate and draw necks
    neck_thickness_45_135, neck_x_45_135, neck_y_45_135 = find_neck_thickness(
        plot_x, plot_y, theta, (45, 135)
    )
    neck_thickness_30_150, neck_x_30_150, neck_y_30_150 = find_neck_thickness(
        plot_x, plot_y, theta, (30, 150)
    )

    # Draw neck lines
    ax_plot.plot(
        [neck_x_45_135, neck_x_45_135],
        [-neck_thickness_45_135 / 2, neck_thickness_45_135 / 2],
        color='green',
        linewidth=2,
        label='Neck (45-135)'
    )
    ax_plot.plot(
        [neck_x_30_150, neck_x_30_150],
        [-neck_thickness_30_150 / 2, neck_thickness_30_150 / 2],
        color='purple',
        linewidth=2,
        label='Neck (30-150)'
    )

    # Set plot properties
    max_radius = np.max(np.abs(plot_radius)) * 1.5
    ax_plot.set_xlim(-max_radius, max_radius)
    ax_plot.set_ylim(-max_radius, max_radius)
    ax_plot.set_aspect('equal')
    ax_plot.grid(True)
    ax_plot.set_title(f'Nuclear Shape - {event.event_category}', fontsize=18)
    ax_plot.set_xlabel('X (fm)', fontsize=18)
    ax_plot.set_ylabel('Y (fm)', fontsize=18)
    ax_plot.legend(fontsize='small', loc='upper right')

    # Calculate measurements
    max_x_length = np.max(plot_y) - np.min(plot_y)
    max_y_length = np.max(plot_x) - np.min(plot_x)
    along_x_length = (calculate_radius(0.0, full_beta_parameters, event.number_of_protons, event.number_of_neutrons) +
                      calculate_radius(np.pi, full_beta_parameters, event.number_of_protons, event.number_of_neutrons))
    along_y_length = (calculate_radius(np.pi / 2, full_beta_parameters, event.number_of_protons, event.number_of_neutrons) +
                      calculate_radius(-np.pi / 2, full_beta_parameters, event.number_of_protons, event.number_of_neutrons))

    # Calculate volumes
    sphere_volume = calculate_sphere_volume(event.number_of_protons, event.number_of_neutrons)
    shape_volume = calculate_volume(event.number_of_protons, event.number_of_neutrons, full_beta_parameters)
    volume_fix = calculate_volume_fixing_factor(event.number_of_protons, event.number_of_neutrons, full_beta_parameters)

    # Check volume calculation and negative radius
    shape_volume_integration = calculate_volume_by_integration(
        event.number_of_protons, event.number_of_neutrons, full_beta_parameters
    )
    volume_mismatch = abs(sphere_volume - shape_volume_integration) > 1.0
    negative_radius = np.any(plot_radius < 0)

    # Add text information
    info_text = (
        f'Protons: {file_params["number_of_protons"]}, Neutrons: {file_params["number_of_neutrons"]}\n'
        f'Excitation Energy: {file_params["excitation_energy"]} MeV, Angular Momentum: {file_params["angular_momentum"]}ℏ\n'
        f'Event Category: {event.event_category}\n'
        f'Step Count: {event.step_count}\n\n'
        f'Mass: {event.mass:.4f}\n'
        f'Energies (MeV):\n'
        f'Total: {event.total_energy:.4f}\n'
        f'Liquid Drop: {event.liquid_drop_energy:.4f}\n'
        f'Shell Correction: {event.shell_correction_energy:.4f}\n'
        f'Rotational: {event.rotational_energy:.4f}\n\n'
        f'Deformation Parameters:\n'
        f'β10: {event.beta_parameters[0]:.4f}\n'
        f'β20: {event.beta_parameters[1]:.4f}\n'
        f'β30: {event.beta_parameters[2]:.4f}\n'
        f'β40: {event.beta_parameters[3]:.4f}\n'
        f'β50: {event.beta_parameters[4]:.4f}\n'
        f'β60: {event.beta_parameters[5]:.4f}\n'
        f'β70: 0.0000\n'
        f'β80: 0.0000\n\n'
        f'Fission Parameters:\n'
        f'Theta: {event.fission_theta:.4f}°\n'
        f'Ratio: {event.fission_ratio:.4f}\n\n'
        f'Shape Measurements:\n'
        f'Sphere Volume: {sphere_volume:.4f} fm³\n'
        f'Shape Volume: {shape_volume:.4f} fm³\n'
        f'Volume Fixing Factor: {volume_fix:.4f}\n'
        f'Radius Fixing Factor: {volume_fix ** (1 / 3):.4f}\n'
        f'Max X Length: {max_x_length:.2f} fm\n'
        f'Max Y Length: {max_y_length:.2f} fm\n'
        f'Length Along X Axis: {along_x_length:.2f} fm\n'
        f'Length Along Y Axis: {along_y_length:.2f} fm\n'
        f'Neck (45°-135°): {neck_thickness_45_135:.2f} fm\n'
        f'Neck (30°-150°): {neck_thickness_30_150:.2f} fm\n'
    )

    if negative_radius:
        info_text += '\nWarning: Negative radius detected!'
    if volume_mismatch:
        info_text += f'\nWarning: Volume mismatch detected!\n({sphere_volume:.4f} vs {shape_volume_integration:.4f} fm³)'

    ax_text.text(0.1, 0.95, info_text, fontsize=12, verticalalignment='top')

    # Save the plot with event number in filename
    baseFilename = output_filename.rsplit('.', 1)[0]
    newFilename = f"{event_number:d}_{baseFilename}.png"
    plt.savefig(newFilename, dpi=600, bbox_inches='tight')
    plt.close(fig)


def parse_filename(filename):
    """
    Parse the input filename to extract nuclear parameters.

    Args:
    filename (str): Input filename in format "Z_N_E_J_xxxxx_FG_x.x_Endpoints.txt"

    Returns:
    dict: Dictionary containing the parsed parameters
    """
    parts = filename.split('_')
    return {
        'number_of_protons': int(parts[0]),
        'number_of_neutrons': int(parts[1]),
        'excitation_energy': float(parts[2]),
        'angular_momentum': int(parts[3])
    }


def main():
    """Main function to read parameters and generate plots."""
    if len(sys.argv) != 2:
        print("Usage: python ShapeReader.py <input_file>")
        sys.exit(1)

    inputFile = sys.argv[1]
    filesCreated = 0
    maxFiles = 10

    try:
        with open(inputFile) as f:
            for lineNumber, line in enumerate(f, 1):
                try:
                    # Parse the event data
                    event = parse_line(line)

                    # Generate output filename using only the available beta parameters
                    beta_values = "_".join(f"{p:.2f}" for p in event.beta_parameters)
                    outputFilename = (
                        f"{event.event_category}_{event.number_of_protons}_"
                        f"{event.number_of_neutrons}_{event.step_count}_{beta_values}.png"
                    )

                    print(f"Processing line {lineNumber}: {event.event_category}, "
                          f"Z={event.number_of_protons}, N={event.number_of_protons}, "
                          f"Step={event.step_count}")

                    create_plot(event, outputFilename, lineNumber, inputFile)
                    print(f"Saved plot as {lineNumber:01d}_{outputFilename}")

                    filesCreated += 1
                    if filesCreated >= maxFiles:
                        print(f"\nReached limit of {maxFiles} files. Stopping.")
                        break

                except ValueError as e:
                    print(f"Error processing line {lineNumber}: {e}")
                    continue

    except FileNotFoundError:
        print(f"Error: Input file '{inputFile}' not found")
        sys.exit(1)


if __name__ == '__main__':
    main()
