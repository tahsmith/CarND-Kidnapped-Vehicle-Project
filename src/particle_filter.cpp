/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <iostream>
#include <sstream>
#include <cassert>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    double std_x = 2;
    double std_y = 2;
    double std_theta = 0.05;
    std::default_random_engine gen;
    using normal_dist = std::normal_distribution<double>;
    normal_dist x_dist(x, std_x);
    normal_dist y_dist(y, std_y);
    normal_dist theta_dist(theta, std_theta);

    particles.reserve(num_particles);
    for(size_t i = 0; i < num_particles; ++i)
    {
        particles.emplace_back();
        auto& particle = particles.back();
        particle.id = int(i);
        particle.x = x_dist(gen);
        particle.y = y_dist(gen);
        particle.theta = theta_dist(gen);
        particle.weight = 1.0;
        weights.push_back(particle.weight);
    }
    is_initialized = true;
    assert(particles.size() == num_particles);
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    if (fabs(velocity) > 0.001)
    {
        double c = velocity / yaw_rate;
        for (auto& particle : particles)
        {
            double theta0 = particle.theta;
            double theta1 = theta0 + delta_t * yaw_rate;
            particle.x += c * (sin(theta1) - sin(theta0));
            particle.y += c * (cos(theta0) - cos(theta1));
            particle.theta = theta1;
        }
    }
    else
    {
        for (auto& particle : particles)
        {
            double theta = particle.theta;
            particle.x += cos(theta);
            particle.y += sin(theta);
            particle.theta = theta;
        }
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (size_t observation_i = 0;
         observation_i < observations.size(); ++observation_i)
    {
        double best_distance = std::numeric_limits<double>::infinity();
        size_t best_match_i = 0;

        for (size_t predicted_i = 0;
             predicted_i < predicted.size(); ++predicted_i)
        {
            double distance = dist(
                predicted[predicted_i].x, predicted[predicted_i].y,
                observations[observation_i].x, observations[observation_i].y
            );

            if (distance < best_distance)
            {
                best_distance = distance;
                best_match_i = observation_i;
            }
        }
        observations[observation_i].id = predicted[best_match_i].id;
    }

}

LandmarkObs transformGlobalToLocal(const Map::single_landmark_s& globalLandmark, const Particle& particle)
{
    return {
        globalLandmark.id_i,
        0.0,
        0.0
    };
}

std::vector<LandmarkObs> predictParticleObservations(const Map& mapLandmarks, const Particle& particle)
{
    std::vector<LandmarkObs> particleObservations;
    particleObservations.reserve(mapLandmarks.landmark_list.size());
    for (auto&& landmark : mapLandmarks.landmark_list) {
        particleObservations.push_back(transformGlobalToLocal(landmark, particle));
    }
    return particleObservations;
}

std::vector<LandmarkObs> filterPredictedLandmarks(
    const std::vector<LandmarkObs>& observedLandmarks,
    const std::vector<LandmarkObs>& predictedLandmarks)
{
    std::vector<LandmarkObs> filtered;
    filtered.reserve(observedLandmarks.size());
    for (auto&& observed : observedLandmarks) {
        for (auto&& predicted: predictedLandmarks) {
            if (predicted.id == observed.id)
            {
                filtered.push_back(predicted);
                break;
            }
        }
    }
    assert(filtered.size() == observedLandmarks.size());
    return filtered;
}

double observationWeight(const LandmarkObs& predicted, const LandmarkObs& observed, double stdLandmark[2]) {
    double err_x = predicted.x - observed.x;
    double err_y = predicted.y - observed.y;
    return 1 / sqrt(2 * M_PI * stdLandmark[0] * stdLandmark[1])
        * exp( err_x * err_x / stdLandmark[0] + err_y * err_y / stdLandmark[1]);
}

double particleWeight(const Particle& particle, const std::vector<LandmarkObs>& observedLandmarks, const std::vector<LandmarkObs>& predictedLandmarks,  double stdLandmark[2])
{
    double weight = 1;
    assert(observedLandmarks.size() == predictedLandmarks.size());
    for(size_t i = 0; i < observedLandmarks.size(); ++i)
    {
        assert(predictedLandmarks[i].id == observedLandmarks[i].id);
        weight = weight * observationWeight(predictedLandmarks[i], observedLandmarks[i], stdLandmark);
    }
    return weight;
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

    auto mutable_observations = observations;

    for (auto& particle : particles)
    {
        auto predicted_landmarks = predictParticleObservations(map_landmarks, particle);
        dataAssociation(predicted_landmarks, mutable_observations);
        predicted_landmarks = filterPredictedLandmarks(mutable_observations, predicted_landmarks);
        particle.weight = particleWeight(particle, predicted_landmarks, mutable_observations, std_landmark);
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const std::vector<int>& associations,
                                     const std::vector<double>& sense_x,
                                     const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
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
