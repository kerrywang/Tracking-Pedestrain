"""
CS6476 Problem Set 5 imports. Only Numpy and cv2 are allowed.
"""
import numpy as np
import cv2


# Assignment code
class KalmanFilter(object):
    """A Kalman filter tracker"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        """Initializes the Kalman Filter

        Args:
            init_x (int or float): Initial x position.
            init_y (int or float): Initial y position.
            Q (numpy.array): Process noise array.
            R (numpy.array): Measurement noise array.
        """
        self.state = np.array([[init_x], [init_y], [0.], [0.]]) # state
        self.state_transition_matrix = np.asmatrix(np.array([[1, 0, 1, 0],
                                                            [0, 1, 0, 1],
                                                            [0, 0, 1, 0],
                                                            [0, 0, 0, 1]]))
        self.measurable_state = np.asmatrix(np.array([[1, 0, 0, 0],
                                                      [0, 1, 0, 0]]))
        self.state_covariance_matrix = np.asmatrix(np.array([[1, 0, 0, 0],
                                                            [0, 1, 0, 0],
                                                            [0, 0, 1000, 0],
                                                            [0, 0, 0, 1000]]))
        self.process_noise = np.asmatrix(Q)
        self.measurement_noise = np.asmatrix(R)
        self.identity_matrix = np.asmatrix(np.eye(4))

    def predict(self):
        self.state = self.state_transition_matrix * self.state
        self.state_covariance_matrix = self.state_transition_matrix * self.state_covariance_matrix * self.state_transition_matrix.T + self.process_noise


    def correct(self, meas_x, meas_y):
        measurment = np.asmatrix(np.array([[meas_x, meas_y]])).T
        current_state = self.measurable_state * self.state
        error = measurment - current_state

        intermediate = self.measurable_state * self.state_covariance_matrix * self.measurable_state.T + self.measurement_noise
        kalman_gain = self.state_covariance_matrix * self.measurable_state.T * intermediate.I

        self.state = self.state + kalman_gain * error
        self.state_covariance_matrix = (self.identity_matrix - kalman_gain * self.measurable_state) * self.state_covariance_matrix


    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.

    Encapsulating state, initialization and update methods. Refer to
    the method run_particle_filter( ) in experiment.py to understand
    how this class and methods work.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.

        The main components of your particle filter should at least be:
        - self.particles (numpy.array): Here you will store your particles.
                                        This should be a N x 2 array where
                                        N = self.num_particles. This component
                                        is used by the autograder so make sure
                                        you define it appropriately.
                                        Make sure you use (x, y)
        - self.weights (numpy.array): Array of N weights, one for each
                                      particle.
                                      Hint: initialize them with a uniform
                                      normalized distribution (equal weight for
                                      each one). Required by the autograder.
        - self.template (numpy.array): Cropped section of the first video
                                       frame that will be used as the template
                                       to track.
        - self.frame (numpy.array): Current image frame.

        Args:
            frame (numpy.array): color BGR uint8 image of initial video frame,
                                 values in [0, 255].
            template (numpy.array): color BGR uint8 image of patch to track,
                                    values in [0, 255].
            kwargs: keyword arguments needed by particle filter model:
                    - num_particles (int): number of particles.
                    - sigma_exp (float): sigma value used in the similarity
                                         measure.
                    - sigma_dyn (float): sigma value that can be used when
                                         adding gaussian noise to u and v.
                    - template_rect (dict): Template coordinates with x, y,
                                            width, and height values.
        """
        self.num_particles = kwargs.get('num_particles')  # required by the autograder
        self.sigma_exp = kwargs.get('sigma_exp')  # required by the autograder
        self.sigma_dyn = kwargs.get('sigma_dyn')  # required by the autograder
        self.template_rect = kwargs.get('template_coords')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

        self.template = template
        self.frame = frame
        self.particles = np.zeros((self.num_particles, 2))  # Initialize your particles array. Read the docstring.
        self.weights = np.array(np.ones((self.num_particles), dtype=np.float32)) / self.num_particles  # Initialize your weights array. Read the docstring.
        # Initialize any other components you may need when designing your filter.

        # randomly distribute particles accross the map
        self.particles[:, 0] = np.random.uniform(self.template_rect['x'], self.template_rect['x'] + self.template_rect['w'], self.num_particles)
        self.particles[:, 1] = np.random.uniform(self.template_rect['y'], self.template_rect['y'] + self.template_rect['h'], self.num_particles)

        self.grayed_template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)


    def get_particles(self):
        """Returns the current particles state.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: particles data structure.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.

        This method is used by the autograder. Do not modify this function.

        Returns:
            numpy.array: weights data structure.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.

        Returns:
            float: similarity value.
        """
        if template.shape != frame_cutout.shape:
            return 0.
        mse = np.sum((template.astype('float') - frame_cutout.astype('float'))**2) / (template.shape[0] * template.shape[1])
        return np.exp(-mse / (2 * self.sigma_exp ** 2))

    def resample_particles(self):
        """Returns a new set of particles

        This method does not alter self.particles.

        Use self.num_particles and self.weights to return an array of
        resampled particles based on their weights.

        See np.random.choice or np.random.multinomial.
        
        Returns:
            numpy.array: particles data structure.
        """
        resampled_index = np.random.choice(np.arange(self.num_particles), self.num_particles, True, p=self.weights)
        return self.particles[resampled_index]

    def obtain_template(self, gray_frame, center):
        height, width = self.template.shape[0], self.template.shape[1]
        start_x, end_x = np.clip(center[0] - width / 2, 0, gray_frame.shape[1] - 1), np.clip(
            center[0] + width / 2, 0, gray_frame.shape[1] - 1)
        start_y, end_y = np.clip(center[1] - height / 2, 0, gray_frame.shape[0] - 1), np.clip(
            center[1] + height / 2, 0, gray_frame.shape[0] - 1)

        target_frame = gray_frame[int(start_y): int(end_y), int(start_x): int(end_x)]
        return target_frame

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        Implement the particle filter in this method returning None
        (do not include a return call). This function should update the
        particles and weights data structures.

        Make sure your particle filter is able to cover the entire area of the
        image. This means you should address particles that are close to the
        image borders.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_particles = self.resample_particles()
        # move the particle with gaussian noise
        new_particles[:, 0] += np.random.normal(0, self.sigma_dyn, self.num_particles)
        new_particles[:, 1] += np.random.normal(0, self.sigma_dyn, self.num_particles)

        new_particles[:, 0] = np.clip(new_particles[:, 0], 0, self.frame.shape[0] - 1)
        new_particles[:, 1] = np.clip(new_particles[:, 1], 0, self.frame.shape[1] - 1)
        self.particles = new_particles

        new_weights = np.zeros_like(self.weights, dtype=np.float32)

        for i in range(self.particles.shape[0]):
            target_frame = self.obtain_template(gray_frame, self.particles[i])
            new_weight = self.get_error_metric(self.grayed_template, target_frame)
            new_weights[i] = new_weight

        self.weights = new_weights / np.sum(new_weights)


    def render(self, frame_in):
        """Visualizes current particle filter state.

        This method may not be called for all frames, so don't do any model
        updates here!

        These steps will calculate the weighted mean. The resulting values
        should represent the tracking window center point.

        In order to visualize the tracker's behavior you will need to overlay
        each successive frame with the following elements:

        - Every particle's (x, y) location in the distribution should be
          plotted by drawing a colored dot point on the image. Remember that
          this should be the center of the window, not the corner.
        - Draw the rectangle of the tracking window associated with the
          Bayesian estimate for the current location which is simply the
          weighted mean of the (x, y) of the particles.
        - Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius.

        This function should work for all particle filters in this problem set.

        Args:
            frame_in (numpy.array): copy of frame to render the state of the
                                    particle filter.
        """

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
            x_weighted_mean += self.particles[i, 0] * self.weights[i]
            y_weighted_mean += self.particles[i, 1] * self.weights[i]

            cv2.circle(frame_in, tuple(map(int, self.particles[i][:2])), 1, (0, 0, 255), -1)

        top_left_y = int(y_weighted_mean - self.grayed_template.shape[0] / 2)
        top_left_x = int(x_weighted_mean - self.grayed_template.shape[1] / 2)

        bot_right_y = int(y_weighted_mean + self.grayed_template.shape[0] / 2)
        bot_right_x = int(x_weighted_mean + self.grayed_template.shape[1] / 2)

        cv2.rectangle(frame_in, (top_left_x, top_left_y), (bot_right_x, bot_right_y), (0, 255, 255), 2)

        """Finally we need to get some sense of the standard deviation or
          spread of the distribution. First, find the distance of every
          particle to the weighted mean. Next, take the weighted sum of these
          distances and plot a circle centered at the weighted mean with this
          radius."""

        dist = self.particles[:, :2] - np.array((x_weighted_mean, y_weighted_mean))
        dist = dist[:, 0] ** 2 + dist[:, 1] ** 2
        weighted_sum = np.sum(dist * self.weights)

        cv2.circle(frame_in, (int(x_weighted_mean), int(y_weighted_mean)), int(weighted_sum), (255, 255, 0), 2)

class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):
        """Initializes the appearance model particle filter.

        The documentation for this class is the same as the ParticleFilter
        above. There is one element that is added called alpha which is
        explained in the problem set documentation. By calling super(...) all
        the elements used in ParticleFilter will be inherited so you do not
        have to declare them again.
        """

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor

        self.alpha = kwargs.get('alpha')  # required by the autograder
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "Appearance Model" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame, values in [0, 255].

        Returns:
            None.
        """
        super(AppearanceModelPF, self).process(frame)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        idx = np.average(self.particles, axis=0, weights=self.weights).astype(np.int)
        best_template = self.obtain_template(gray_frame, idx)

        self.grayed_template = (self.alpha * best_template + (1 - self.alpha) * self.grayed_template).astype(np.int)





class MDParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        """Initializes MD particle filter object.

        The documentation for this class is the same as the ParticleFilter
        above. By calling super(...) all the elements used in ParticleFilter
        will be inherited so you don't have to declare them again.
        """

        super(MDParticleFilter, self).__init__(frame, template, **kwargs)  # call base class constructor
        # If you want to add more parameters, make sure you set a default value so that
        # your test doesn't fail the autograder because of an unknown or None value.
        #
        # The way to do it is:
        # self.some_parameter_name = kwargs.get('parameter_name', default_value)

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.

        This process is also inherited from ParticleFilter. Depending on your
        implementation, you may comment out this function and use helper
        methods that implement the "More Dynamics" procedure.

        Args:
            frame (numpy.array): color BGR uint8 image of current video frame,
                                 values in [0, 255].

        Returns:
            None.
        """
        super(MDParticleFilter, self).process(frame)