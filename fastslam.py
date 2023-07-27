import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import time

def read_landmarks(filename):
    """
    Read landmark positions from a file
    :param filename: name of the file
    :return: array of landmark positions [[x, y]]
    """
    return np.loadtxt(filename)[:, 1:]


def read_sensor_data(filename):
    """
    Read sensor data from a file
    :param filename: name of the file
    :return: a dictionary of odometry and sensor data, 
    odometry: [[d_r1, d_t, d_r2]], sensor: [[[id, range, bearing]]]
    """
    df = pd.read_csv(filename, sep=" ", header=None)
    odom_idxs = np.where(df[0] == "ODOMETRY")[0]

    data = {"odometry": [], "sensor": []}
    for i in range(len(odom_idxs)):
        odometry = df.iloc[odom_idxs[i], 1:].to_numpy(dtype=np.float64)

        if i == len(odom_idxs) - 1:
            sensor = df.iloc[odom_idxs[i]+1:, 1:].to_numpy(dtype=np.float64)
        else:
            sensor = df.iloc[odom_idxs[i]+1:odom_idxs[i+1], 1:].to_numpy(dtype=np.float64)

        data["odometry"].append(odometry)
        data["sensor"].append(sensor)

    data["odometry"] = np.array(data["odometry"])
    data["sensor"] = np.array(data["sensor"], dtype=object)
    return data


def measurement_model(particle, landmark_mu):
    """Compute the expected measurement h and Jacobian H for each landmark.
    :param particle: a single particle in form [x, y, theta, weight]
    :param landmark_mu: a set of landmark positions of shape [n_landmarks, 2]
    :return measurement model h [n_landmarks, 2], Jacobian of the measurement model H [n_landmarks, 2, 2]
    """
    # Break data appart for readability
    x, y, theta, _ = particle
    lx, ly = landmark_mu[:, 0], landmark_mu[:, 1]

    # Compute the expected range and bearing h [n_landmarks, 2]
    h = np.array([
        np.sqrt((lx - x)**2 + (ly - y)**2),
        np.arctan2(ly - y, lx - x) - theta
    ]).T

    # Compute the Jacobian H [n_landmarks, 2, 2] wrt landmark position
    H = np.zeros((lx.shape[0], 2, 2))
    H[:, 0, 0] = (lx - x) / h[:, 0]
    H[:, 0, 1] = (ly - y) / h[:, 0]
    H[:, 1, 0] = (y - ly) / (h[:, 0]**2)
    H[:, 1, 1] = (lx - x) / (h[:, 0]**2)
    return h, H

def angle_diff(angle1, angle2):
    """Difference between angles."""
    return np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))


class SLAMParticles:
    """Wrapper for the particles and their estimate of landmarks."""
    def __init__(self, n_particles, n_landmarks):
        # Particles in form [x, y, theta, weight]
        self.p = np.zeros((n_particles, 4))
        self.p[:,-1] = 1.0 / n_particles
        # Position of landmarks, each particle holds a set of landmarks in form [[l_x, l_y], ...]
        self.l_mu = np.zeros((n_particles, n_landmarks, 2))
        # Covariance of landmark poses, each particle holds a set of landmarks in form [[[s_xx, s_xy], [s_yx, s_yy]], ...]
        self.l_sigma = np.zeros((n_particles, n_landmarks, 2, 2))
        # For each particle store whether a particular landmark was observed or not
        self.l_observed = np.zeros((n_particles, n_landmarks), dtype=bool)
        
        
class FastSLAM:
    """Landmark based FastSLAM"""
    def __init__(self, n_particles, n_landmarks, range_var, bearing_var, alpha):
        """
        :param n_particles: Number of particles used to estimate the pose of the robot and position of landmarks
        :param n_landmarks: Number of landmarks to estimate
        :param range_var: Variance associated with range measurements
        :param bearing_var: Variance associated with bearing measurements
        """
        self.n_particles = n_particles
        self.n_landmarks = n_landmarks
        self.Q_t = np.array([[range_var, 0], [0, bearing_var]]) # sensor noise
        self.alpha = alpha
    
        self.particles = SLAMParticles(self.n_particles, self.n_landmarks)
        
    def __call__(self, odometry_data, sensor_data):
        """Estimate the pose of the robot and the position of landmarks."""
        self._sample_motion_model(odometry_data)
        self._eval_sensor_model(sensor_data)
        self._resample_particles()
        return self._best_pose()
        
    def _sample_motion_model(self, odometry_data):
        """Predict the new pose of the target given the odometry_data"""
        d_r1, d_t, d_r2 = odometry_data
        a1, a2, a3, a4 = self.alpha
        
        # Add noise to odometry data
        d_r1_hat = d_r1 + np.random.randn(self.n_particles) * (a1*abs(d_r1) + a2*abs(d_t))
        d_r2_hat = d_r2 + np.random.randn(self.n_particles) * (a1*abs(d_r2) + a2*abs(d_t))
        d_t_hat = d_t + np.random.randn(self.n_particles)  * (a3*abs(d_t) + a4*(abs(d_r1) + abs(d_r2)))
        
        # Update pose of the particles, using the noisy odometry data
        self.particles.p[:, 0] += d_t_hat * np.cos(self.particles.p[:, 2] + d_r1_hat)
        self.particles.p[:, 1] += d_t_hat * np.sin(self.particles.p[:, 2] + d_r1_hat)
        self.particles.p[:, 2] += d_r1_hat + d_r2_hat
        
    def _eval_sensor_model(self, sensor_data):
        """Correct the position of landmarks, and update particle imporatance"""
        observed_landmark_idxs = sensor_data[:, 0].astype(int) - 1
        range_meas = sensor_data[:, 1]
        bearing_meas = sensor_data[:, 2]

        for idx in range(self.n_particles):
            # Fetch the observed boolean flag array for the current particle
            l_observed = self.particles.l_observed[idx, observed_landmark_idxs]
            seen, not_seen = (l_observed == True), (l_observed == False)
            idx_seen, idx_not_seen = observed_landmark_idxs[seen], observed_landmark_idxs[not_seen]

            # Initialize the position estimate of the not (yet) seen landmarks
            if len(idx_not_seen) > 0:
                _, H = measurement_model(self.particles.p[idx], self.particles.l_mu[idx, idx_not_seen])

                # Estimate the position of landmarks
                delta_pos = np.array([
                    range_meas[not_seen] * np.cos(bearing_meas[not_seen] + self.particles.p[idx, 2]),
                    range_meas[not_seen] * np.sin(bearing_meas[not_seen] + self.particles.p[idx, 2]),
                ]).T # shape: [n_not_seen, 2]
                self.particles.l_mu[idx, idx_not_seen] = self.particles.p[idx, :2][np.newaxis, :] + delta_pos
                
                # Estimate the covariance matrix associated with the estiamted positon
                H_inv = np.linalg.inv(H)
                self.particles.l_sigma[idx, idx_not_seen, :, :] = H_inv @ self.Q_t[np.newaxis, :, :] @ np.transpose(H_inv, (0, 2, 1))
                self.particles.l_observed[idx, idx_not_seen] = True
            
            # Update the position estimate of landmarks
            if len(idx_seen) > 0:
                h, H = measurement_model(self.particles.p[idx], self.particles.l_mu[idx, idx_seen])
                Q = H @ self.particles.l_sigma[idx, idx_seen, :, :] @ np.transpose(H, (0, 2, 1)) + self.Q_t[np.newaxis, :, :]
                K = self.particles.l_sigma[idx, idx_seen, :, :] @ np.transpose(H, (0, 2, 1)) @ np.linalg.inv(Q)

                delta = np.array([
                    range_meas[seen] - h[:, 0], 
                    angle_diff(bearing_meas[seen], h[:, 1].flatten())
                ]).T # shape: [n_seen, 2]

                self.particles.l_mu[idx, idx_seen] += (K @ delta[:, :, np.newaxis]).squeeze()
                I = np.tile(np.eye(2), (idx_seen.shape[0], 1, 1))
                self.particles.l_sigma[idx, idx_seen] = (I - K @ H) @ self.particles.l_sigma[idx, idx_seen, :, :]

                # Update the weight
                self.particles.p[idx, -1] = np.linalg.det(2*np.pi*Q)**(-0.5) @ \
                    np.exp(-0.5 * (np.transpose(delta[:, :, np.newaxis], (0, 2, 1)) @ np.linalg.inv(Q) @ delta[:, :, np.newaxis]).flatten())
        
        # Normalize weights
        self.particles.p[:, 3] /= self.particles.p[:, 3].sum()

    def _resample_particles(self):
        """Resample slam particles using stochastic universal sampling"""
        # compute the cdf
        sampled_idxs = []
        c_i = self.particles.p[:, -1][0]

        # initialize threshold
        u, i = 1 / self.n_particles * np.random.rand(), 0
        for _ in range(0, self.n_particles):
            # skip until the threshold is reached
            while u > c_i: 
                i += 1
                c_i += self.particles.p[:, -1][i]

            # increment the threshold and store the particle
            sampled_idxs.append(i)
            u = u + 1 / self.n_particles

        assert len(sampled_idxs) == self.n_particles, f"{len(sampled_idxs)} != {self.n_particles} resampling failed!"

        # Update the set of slam particles
        self.particles.p = self.particles.p[sampled_idxs]
        self.particles.l_mu = self.particles.l_mu[sampled_idxs]
        self.particles.l_sigma = self.particles.l_sigma[sampled_idxs]
        self.particles.l_observed = self.particles.l_observed[sampled_idxs]

    def _best_pose(self):
        """Compute the best pose of a particle set."""
        best_idx = np.argmax(self.particles.p[:, -1])
        return self.particles.p[best_idx], self.particles.l_mu[best_idx], self.particles.l_sigma[best_idx] 


def error_ellipse(l_mu, l_sigma, chisquare_scale=2.2789, color="red"):
    """Compute the ellipse, which estimates the position of the landmark."""
    eigen_vals, eigen_vectors = np.linalg.eig(l_sigma)
    # Get the largest eigen-vector
    max_idx = np.argmax(eigen_vals)
    max_eigen_vect = eigen_vectors[:, max_idx]
    max_eigen_val = eigen_vals[max_idx]

    # Get the smallest eigen-vector
    min_idx = np.argmin(eigen_vals)
    min_eigen_vect = eigen_vectors[:, min_idx]
    min_eigen_val = eigen_vals[min_idx]

    # Calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chisquare_scale*max_eigen_val)
    height = 2 * np.sqrt(chisquare_scale*min_eigen_val)
    angle = np.arctan2(min_eigen_vect[1], min_eigen_vect[0])

    # Generate covariance ellipse
    error_ellipse = Ellipse(
        xy=[l_mu[0],l_mu[1]], width=width, height=height, angle=angle/np.pi*180, color=color)
    error_ellipse.set_alpha(0.25)
    return error_ellipse


fig, ax = plt.subplots()
def init():
    ax.set_title("Landmark based FastSLAM")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend()
    return ax,


def simulate(idx):
    odom, sens = sensor_data["odometry"][idx], sensor_data["sensor"][idx]
    estimated_pose, l_mu, l_sigma = slam(odom, sens)
    plt.clf()
    plt.gca().set(frame_on=False)
    plt.title("Landmark based FastSLAM")
    plt.quiver(slam.particles.p[:, 0], slam.particles.p[:, 1], np.cos(slam.particles.p[:, 2]),
               np.sin(slam.particles.p[:, 2]), angles='xy', scale_units='xy', color="#3776ab")
    plt.scatter(x=landmarks[:, 0], y=landmarks[:, 1], color='black', label="landmarks", marker="o", s=60)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]),
               np.sin(estimated_pose[2]), angles='xy', scale_units='xy', color="gray", label="estimated pose")

    # plot the estimated position of the landmarks
    for idx in range(len(landmarks)):
        plt.gca().add_artist(error_ellipse(l_mu[idx], l_sigma[idx]))

    plt.axis(map_limits)
    plt.xticks([]); plt.yticks([])
    plt.legend()
    return ax, 


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("--n_particles", type=int, default=100, help="number of particles")
    argparse.add_argument("--sigma_range", type=float, default=0.1, help="standard deviation of the range measurement")
    argparse.add_argument("--sigma_bearing", type=float, default=0.1, help="standard deviation of the bearing measurement")
    argparse.add_argument("--world", type=str, default="./world.dat", help="world file")
    argparse.add_argument("--sensor_data", type=str, default="./sensor_data.dat", help="sensor data file")
    args = argparse.parse_args()

    map_limits = [-1, 12, 0, 10]
    # Read the data
    landmarks = read_landmarks(args.world)
    sensor_data = read_sensor_data(args.sensor_data)

    # Initi the Fast SLAM model
    slam = FastSLAM(n_particles=args.n_particles, n_landmarks=landmarks.shape[0],
                    range_var=args.sigma_range, bearing_var=args.sigma_bearing, alpha=[0.1, 0.1, 0.05, 0.05])
    
    # Similate the SLAM
    print("Simulating the FastSLAM...")
    sim = FuncAnimation(fig, simulate, init_func=init, frames=len(sensor_data["odometry"]), interval=20, blit=True)
    sim.save('fastslam_sim.gif', writer='imagemagick')
