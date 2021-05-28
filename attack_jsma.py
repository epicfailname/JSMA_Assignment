import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from pixelmap import (AlgorithmEnum, Homography)
from utilities import (generate_hull_mask, plot_and_save_graph)

TF_DTYPE = tf.as_dtype('float32')
CLIP_MAX = 1.0
CLIP_MIN = 0.0
MAX_ITERATIONS = 50

class JSMARegressionAttack:
    def __init__(self,
            model,
            resultsdir,
            max_iters=MAX_ITERATIONS,
            pixelmap_algo=None,
            debug_flag=True,
            clip_max=CLIP_MAX,
            clip_min=CLIP_MIN,
            increase=True,
            is_mask=True):
        """
        The JSMA attack.
        Returns adversarial examples for the supplied model.
        model: The model on which we perform the attack on.
        max_iters: The maximum number of iterations.
          Corresponds to the number of (pixel, colour) coordinates to perturb
        pixelmap_algo: Which mapping algorithm to use for the pixel mapping (don't map if None)
        debug_flag: Flag to print debug statements.
        clip_max: Maximum pixel value (default 1.0).
        clip_min: Minimum pixel value (default 0.0).
        increase: Direction of pixel values to perturb towards
          Does NOT affect the adversarial steering angles
        is_mask: Flag; Do we constraint the pertubation to a speciific region?
        """

        self.model = model
        self.resultsdir = resultsdir
        self.max_iters = max_iters
        self.pixelmap_algo = pixelmap_algo
        self.debug_flag = debug_flag
        
        self.clip_max = clip_max
        self.clip_min = clip_min
        self.increase = increase
        self.is_mask = is_mask


    def diff_avg(self, adv_diffs):
        return sum(adv_diffs) / len(adv_diffs)

    def seq_diff_avg(self, adv_diffs):
        adv_diffs_copy = adv_diffs.copy()
        seq_diffs = list(map(lambda pair: abs(pair[0] - pair[1]), zip(adv_diffs[1:], adv_diffs[:len(adv_diffs) - 1])))
        return sum(seq_diffs) / len(seq_diffs)

    def debug(self, *args):
        padding = '\t'
        if self.debug_flag:
            print('DEBUG:   ' + padding.join(map(str, args[0:])))

    def attack(self, data):
        # Sanity checks on data
        imgs, targets = data.input_data, data.output_data
        assert(len(imgs) == len(targets))
        if self.is_mask:
            sequence_of_list_of_corners = data.sequence_of_list_of_corners
            assert(len(sequence_of_list_of_corners) == len(imgs))

        print('Number of attack targets:    ', len(imgs))
        if self.is_mask:
            if self.pixelmap_algo:
                return self.attack_pixelmap(data)
            else:
                return self.attack_batch(data)
        else:
            raise NotImplementedError('Not required for this assignment')

    def attack_batch(self, data):
        """
        Run the attack on a batch of images and labels.
        """
        imgs = data.input_data
        labs = data.output_data
        batch_size = len(imgs)
        sequence_of_list_of_corners = data.sequence_of_list_of_corners

        x = tf.cast(tf.constant(imgs), tf.float32)
        y_true = tf.expand_dims(tf.cast(tf.constant(labs), tf.float32), axis = 1)
        self.debug('img Shape:', imgs.shape)
        self.debug('lab:', labs, labs.shape)
        self.debug('x Shape:', x.shape, x.dtype)
        self.debug('y_true Shape:', y_true.shape, y_true.dtype)

        # Compute our initial search domain.  We optimize the initial search domain
        # by removing all features that are already at their maximum values (if
        # increasing input features---otherwise, at their minimum value).
        if self.increase:
            search_domains = tf.Variable(tf.reshape(tf.equal(tf.cast(x < self.clip_max, TF_DTYPE), 1.0), x.shape))
        else:
            search_domains = tf.Variable(tf.reshape(tf.equal(tf.cast(x > self.clip_min, TF_DTYPE), 1.0), x.shape))

        # Manually calculate the allowed perturbation area
        # search_domains: A boolean mask to apply over the input images
        # search_domain_vertices: A list (tensor) of vertices of the input images that are allowed to be perturbed
        if self.is_mask:
            for i in range(batch_size):
                list_of_corners = sequence_of_list_of_corners[i]
                pertubation_mask = generate_hull_mask(x.shape[1:3], np.array(list_of_corners))
                search_domains = search_domains[i].assign(tf.math.logical_and(search_domains[i], tf.expand_dims(pertubation_mask, axis = 2)))
        sparse_search_domains = tf.sparse.from_dense(tf.cast(tf.where(search_domains, x = 1., y = 0.), dtype = TF_DTYPE))
        search_domain_vertices = sparse_search_domains.indices
        self.debug('Search domain shape:', search_domains.shape, search_domains.dtype)
        self.debug('No. of allowed vertices to be perturb:', (search_domain_vertices.shape[0]))

        # Create the variable tensor to calculate the jacobian
        deltas = tf.Variable(initial_value = tf.zeros_like(x),
            shape = x.shape,
            name = 'deltas',
            dtype = TF_DTYPE)
        self.debug('Delta Shape:', deltas.shape, deltas.dtype)

        original_pred = self.model.predict(x)
        self.debug('Original Prediction:', original_pred)

        # List of adversarial predictions and difference against original predictions in each iteration
        adv_preds = []
        adv_diffs = []
        adv_preds.append(original_pred)
        adv_diffs.append(tf.zeros(shape = [batch_size, 1]))

        TIME_START = time.process_time()
        for i in range(self.max_iters):
            print('Rounds of perturbation:  ', i + 1)
            TIME_ROUND_START = time.process_time()

            # Construct the computation graph to calculate the gradients (Jacobians)
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(deltas)
                y = self.model(x + deltas)
                mseloss = (y - y_true) * (y - y_true) * 1 / 2

            # Will have shape: (batch size, <image.shape>)
            jacs = tf.squeeze(tape.batch_jacobian(mseloss, deltas), axis = 1)
            self.debug('Jacobian Shape:', jacs.shape, jacs.dtype)

            to_add = tf.Variable(tf.zeros_like(deltas, dtype = TF_DTYPE))

            # TODO
            # 1) Use `search_domains` and/or `search_domain_vertices` to find the next pixel to update
            # 2) Assign `to_add`, and keep track of pixels updated so you don't pick it again in the next iteration


            # End of TODO

            # Update deltas
            deltas.assign(deltas + to_add)

            adv_pred = self.model.predict(x + deltas)
            self.debug('Adversarial Prediction:', adv_pred)
            self.debug('Adversarial Prediction Difference:', original_pred - adv_pred)

            TIME_END = time.process_time()
            print('Iteration Time Elapsed:  ', TIME_END - TIME_ROUND_START)
            TIME_ROUND_START= TIME_END
            print('Total Time Elapsed:      ', TIME_END - TIME_START)

            # Record the effectiveness of perturbation at this iteration
            adv_preds.append(adv_pred)
            adv_diffs.append(original_pred - adv_pred)

        adv_imgs = imgs + deltas

        adv_preds = tf.squeeze(tf.stack(adv_preds, axis = 1)).numpy()
        with open(os.path.join(self.resultsdir, 'preds.pkl'), 'wb') as f:
            pickle.dump(adv_preds.tolist(), f)
        
        plot_and_save_graph(adv_preds,
            title = 'Predictions_Over_Rounds',
            xlabel = 'Rounds',
            ylabel = 'Adversarial Predictions',
            savedir = self.resultsdir)
        
        adv_diffs = tf.squeeze(tf.stack(adv_diffs, axis = 1)).numpy()
        with open(os.path.join(self.resultsdir, 'adv_diffs.pkl'), 'wb') as f:
            pickle.dump(adv_diffs.tolist(), f)
        
        plot_and_save_graph(adv_preds,
            title = 'Predictions_Difference_Over_Rounds',
            xlabel = 'Rounds',
            ylabel = 'Adversarial Predictions (Difference)',
            savedir = self.resultsdir)
        
        adv_imgs = imgs + deltas
        
        # Save images of the deltas
        for i in range(batch_size):
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(tf.cast(deltas[i] != 0.0, TF_DTYPE), interpolation='none')
            fig.savefig(os.path.join(self.resultsdir, 'delta' + str(i) + '.png'), dpi = 250)
            plt.close(fig)
        
        # Save images of the perturbed target
        for i in range(batch_size):
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(adv_imgs[i], interpolation='none')
            fig.savefig(os.path.join(self.resultsdir, 'adv_image' + str(i) + '.png'), dpi = 250)
            plt.close(fig)

        final_abs_adv_diffs = list(map(lambda diff: abs(diff), tf.squeeze(adv_diffs[:, -1:]).numpy().tolist()))
        print('Average difference:              ', self.diff_avg(final_abs_adv_diffs))
        print('Average sequantial difference:   ', self.seq_diff_avg(final_abs_adv_diffs))
        score = open(os.path.join(self.resultsdir, "score"), "w")
        score.write('Average difference:              ' + str(self.diff_avg(final_abs_adv_diffs)) + '\n')
        score.write('Average sequantial difference:   ' + str(self.seq_diff_avg(final_abs_adv_diffs)) + '\n')
        score.close()

        return adv_imgs

    def attack_pixelmap(self, data):
        """
        Run the attack on a batch of images and labels.
        Do so by pertubring pixels in parallel, mapping pixels using some specified algorithm
        """
        imgs = data.input_data
        labs = data.output_data
        batch_size = len(imgs)
        sequence_of_list_of_corners = data.sequence_of_list_of_corners

        x = tf.cast(tf.constant(imgs), tf.float32)
        y_true = tf.expand_dims(tf.cast(tf.constant(labs), tf.float32), axis = 1)
        self.debug('img Shape:', imgs.shape)
        self.debug('lab:', labs, labs.shape)
        self.debug('x Shape:', x.shape, x.dtype)
        self.debug('y_true Shape:', y_true.shape, y_true.dtype)

        if self.pixelmap_algo == AlgorithmEnum.HOMOGRAPHY:
            for list_of_corners in sequence_of_list_of_corners:
                assert(not list_of_corners is None)
            pixelmap = Homography(sequence_of_list_of_corners, x.shape[1:3])

        # Create the variable tensor to calculate the jacobian
        deltas = tf.Variable(initial_value = tf.zeros_like(x),
            shape = x.shape,
            name = 'deltas',
            dtype = TF_DTYPE)
        self.debug('Delta Shape:', deltas.shape, deltas.dtype)

        original_pred = self.model.predict(x)
        self.debug('Original Pred:', original_pred)

        # List of adversarial predictions and difference against original predictions in each iteration
        adv_preds = []
        adv_diffs = []
        adv_preds.append(original_pred)
        adv_diffs.append(tf.zeros(shape = [batch_size, 1]))

        TIME_START = time.process_time()
        for i in range(self.max_iters):
            # If there are no more vertice strings left to perturbed, end early
            if pixelmap.get_length_of_vertice_strings() == 0:
                break

            #print('Rounds of perturbation:  ', i + 1)
            TIME_ROUND_START = time.process_time()

            # Construct the computation graph to calculate the gradients (Jacobians)
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
                tape.watch(deltas)
                y = self.model(x + deltas)
                mseloss = (y - y_true) * (y - y_true) * 1 / 2

            # Will have shape: (batch size, <image.shape>)
            jacs = tf.squeeze(tape.batch_jacobian(mseloss, deltas), axis = 1)
            self.debug('Jacobian Shape:', jacs.shape, jacs.dtype)

            to_add = tf.Variable(tf.zeros_like(deltas, dtype = TF_DTYPE))

            list_of_vertice_strings = pixelmap.get_list_of_vertice_strings()
            self.debug('Number of vertice strings:', pixelmap.get_length_of_vertice_strings())

            # TODO
            # 1) Use `list_of_vertice_strings` to find the best string of vertices to update in parallel
            # 2) Assign `to_add`, and update `pixelmap` by calling `delete_vertice_string()` so you don't use the same vertice string twice


            # End of TODO

            # Update deltas
            deltas = deltas.assign(deltas + to_add)

            adv_pred = self.model.predict(x + deltas)
            self.debug('Adversarial Prediction:', adv_pred)
            self.debug('Adversarial Prediction Difference:', original_pred - adv_pred)

            #TIME_END = time.process_time()
            #print('Iteration Time Elapsed:  ', TIME_END - TIME_ROUND_START)
            #TIME_ROUND_START= TIME_END
            #print('Total Time Elapsed:      ', TIME_END - TIME_START)

            # Record the effectiveness of perturbation at this iteration
            adv_preds.append(adv_pred)
            adv_diffs.append(original_pred - adv_pred)

        adv_preds = tf.squeeze(tf.stack(adv_preds, axis = 1)).numpy()
        with open(os.path.join(self.resultsdir, 'preds.pkl'), 'wb') as f:
            pickle.dump(adv_preds.tolist(), f)
        
        plot_and_save_graph(adv_preds,
            title = 'Predictions_Over_Rounds',
            xlabel = 'Rounds',
            ylabel = 'Adversarial Predictions',
            savedir = self.resultsdir)
        
        adv_diffs = tf.squeeze(tf.stack(adv_diffs, axis = 1)).numpy()
        with open(os.path.join(self.resultsdir, 'adv_diffs.pkl'), 'wb') as f:
            pickle.dump(adv_diffs.tolist(), f)
        
        plot_and_save_graph(adv_diffs,
            title = 'Predictions_Difference_Over_Rounds',
            xlabel = 'Rounds',
            ylabel = 'Adversarial Predictions (Difference)',
            savedir = self.resultsdir)
        
        adv_imgs = imgs + deltas
        
        # Save images of the deltas
        for i in range(batch_size):
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(tf.cast(deltas[i] != 0.0, TF_DTYPE), interpolation='none')
            fig.savefig(os.path.join(self.resultsdir, 'delta' + str(i) + '.png'), dpi = 250)
            plt.close(fig)
        
        # Save images of the perturbed target
        for i in range(batch_size):
            fig = plt.figure(frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(adv_imgs[i], interpolation='none')
            fig.savefig(os.path.join(self.resultsdir, 'adv_image' + str(i) + '.png'), dpi = 250)
            plt.close(fig)

        final_abs_adv_diffs = list(map(lambda diff: abs(diff), tf.squeeze(adv_diffs[:, -1:]).numpy().tolist()))
        print('Average difference:              ', self.diff_avg(final_abs_adv_diffs))
        print('Average sequantial difference:   ', self.seq_diff_avg(final_abs_adv_diffs))
        score = open(os.path.join(self.resultsdir, "score"), "w")
        score.write('Average difference:              ' + str(self.diff_avg(final_abs_adv_diffs)) + '\n')
        score.write('Average sequantial difference:   ' + str(self.seq_diff_avg(final_abs_adv_diffs)) + '\n')
        score.close()
        
        return adv_imgs

