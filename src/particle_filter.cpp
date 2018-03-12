/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

#define EPS 0.00001

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// Initializing the number of particles.
	num_particles = 100;

	// Extracting standard deviations.
	double std_x 	 = std[0];
	double std_y 	 = std[1];
	double std_theta = std[2];

	// Normal distributions for signal noise.
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	// Generate particles.
	for (int i = 0; i < num_particles; i++) {

		Particle particle;
		particle.id 	= i;
		particle.x 		= dist_x(gen);
		particle.y 		= dist_y(gen);
		particle.theta  = dist_theta(gen);
		particle.weight = 1.0;

		particles.push_back(particle);
	}

	// The filter is now initialized.
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Extracting standard deviations.
	double std_x 	 = std_pos[0];
	double std_y 	 = std_pos[1];
	double std_theta = std_pos[2];

	// Creating normal distributions
	normal_distribution<double> dist_x(0, std_x);
	normal_distribution<double> dist_y(0, std_y);
	normal_distribution<double> dist_theta(0, std_theta);

	// Calculate new state.
	for (int i = 0; i < num_particles; i++) {

		double theta = particles[i].theta;

		if (fabs(yaw_rate) < EPS) {
			// When yaw is unchanged.
			particles[i].x += velocity * delta_t * cos(theta);
			particles[i].y += velocity * delta_t * sin(theta);
		} else {
			particles[i].x 	   += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
			particles[i].y 	   += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}

		// Add noise.
		particles[i].x 	   += dist_x(gen);
		particles[i].y     += dist_y(gen);
		particles[i].theta += dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned int i = 0; i < observations.size(); i++) {

		// Initialize placeholders.
		double min_distance = numeric_limits<double>::max();
		int map_id = -1;

		for (unsigned j = 0; j < predicted.size(); j++) {
			// For each predition.

			double x_distance = observations[i].x - predicted[j].x;
			double y_distance = observations[i].y - predicted[j].y;

			double distance   = x_distance * x_distance + y_distance * y_distance;

			// If the "distance" is less than the minimum, store the id and update the minimum.
			if (distance < min_distance) {
				min_distance = distance;
				map_id 		 = predicted[j].id;
			}
		}

		// Set the observation's id to the nearest predicted landmark's id.
		observations[i].id = map_id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

  	for (int i = 0; i < num_particles; i++) {

		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;

		// Find landmarks in particle's range.
		vector<LandmarkObs> in_range_landmarks;
		for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {

			int landmark_id   = map_landmarks.landmark_list[j].id_i;
			float landmark_x  = map_landmarks.landmark_list[j].x_f;
			float landmark_y  = map_landmarks.landmark_list[j].y_f;
	
			double x_distance = x - landmark_x;
			double y_distance = y - landmark_y;
			double distance   = x_distance * x_distance + y_distance * y_distance;

			if (distance <= sensor_range * sensor_range) {
				in_range_landmarks.push_back(LandmarkObs{ landmark_id, landmark_x, landmark_y });
			}
		}

		// Transform observation coordinates.
		vector<LandmarkObs> mapped_observations;
		for(unsigned int j = 0; j < observations.size(); j++) {

			double observation_x = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
			double observation_y = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
			mapped_observations.push_back(LandmarkObs{ observations[j].id, observation_x, observation_y });
		}

		// Associate the landmarks to the observations for the current particle.
		dataAssociation(in_range_landmarks, mapped_observations);

		// Reset weight.
		particles[i].weight = 1.0;

		// Calculate weights.
		for(unsigned int j = 0; j < mapped_observations.size(); j++) {

			int landmark_id      = mapped_observations[j].id;
			double observation_x = mapped_observations[j].x;
			double observation_y = mapped_observations[j].y;

			double landmark_x, landmark_y;

			// Get the x,y coordinates of the prediction associated with the current observation.
			for (unsigned int k = 0; k < in_range_landmarks.size(); k++) {
				if (in_range_landmarks[k].id == landmark_id) {
					landmark_x = in_range_landmarks[k].x;
					landmark_y = in_range_landmarks[k].y;
					break;
				}
			}
	
			// Calculating weight using multivariate normal distribution.
			double x_distance = observation_x - landmark_x;
			double y_distance = observation_y - landmark_y;

			double std_landmark_x = std_landmark[0];
			double std_landmark_y = std_landmark[1];

			double weight = (1 / (2 * M_PI * std_landmark_x * std_landmark_y)) *
							exp(-((x_distance * x_distance) / (2 * std_landmark_x * std_landmark_x) + 
								 ((y_distance * y_distance) / (2 * std_landmark_y * std_landmark_y))));

			// Product of this observation weight with total observations weight
			particles[i].weight *= weight;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Get weights and max weight.
	vector<double> weights;
	double maxWeight = numeric_limits<double>::min();
	for(int i = 0; i < num_particles; i++) {
		weights.push_back(particles[i].weight);
		if ( particles[i].weight > maxWeight ) {
		maxWeight = particles[i].weight;
		}
	}

	// Creating distributions.
	uniform_real_distribution<double> distDouble(0.0, maxWeight);
	uniform_int_distribution<int> distInt(0, num_particles - 1);

	// Generating index.
	int index = distInt(gen);

	double beta = 0.0;

	// the wheel
	vector<Particle> resampledParticles;
	for(int i = 0; i < num_particles; i++) {
		beta += distDouble(gen) * 2.0;
		while( beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resampledParticles.push_back(particles[index]);
	}

	particles = resampledParticles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
