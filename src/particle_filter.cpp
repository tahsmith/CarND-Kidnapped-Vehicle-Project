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
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];
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
    std::default_random_engine gen;
    using normal_dist = std::normal_distribution<double>;
    normal_dist x_noise(0, std_pos[0]);
    normal_dist y_noise(0, std_pos[1]);
    normal_dist yaw_noise(0, std_pos[2]);
    if (fabs(yaw_rate) > 0.001)
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
        double r = velocity * delta_t;
        for (auto& particle : particles)
        {
            double theta = particle.theta;
            particle.x += r * cos(theta);
            particle.y += r * sin(theta);
            particle.theta = theta;
        }
    }

    for (auto& particle : particles)
    {
        particle.x += x_noise(gen);
        particle.y += y_noise(gen);
        particle.theta += yaw_noise(gen);
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    for (size_t observation_i = 0;
         observation_i < observations.size(); ++observation_i)
    {
        double best_distance = std::numeric_limits<double>::infinity();
        size_t best_match_i = 0;

        for (size_t landmark_i = 0;
             landmark_i < predicted.size(); ++landmark_i)
        {
            double distance = dist(
                predicted[landmark_i].x, predicted[landmark_i].y,
                observations[observation_i].x, observations[observation_i].y
            );

            if (distance < best_distance)
            {
                best_distance = distance;
                best_match_i = landmark_i;
            }
        }
        observations[observation_i].id = predicted[best_match_i].id;
    }

}

LandmarkObs transformLocalToGlobal(const LandmarkObs& observation,
                                   const Particle& particle)
{
    double x = observation.x;
    double y = observation.y;
    double sin_theta = sin(particle.theta);
    double cos_theta = cos(particle.theta);
    return {
        observation.id,
        particle.x + x * cos_theta - y * sin_theta,
        particle.y + x * sin_theta + y * cos_theta,
    };
}

std::vector<LandmarkObs> predictParticleObservations(const std::vector<LandmarkObs>& observations, const Particle& particle)
{
    std::vector<LandmarkObs> particleObservations;
    particleObservations.reserve(observations.size());
    for (auto&& landmark : observations) {
        particleObservations.push_back(
            transformLocalToGlobal(landmark, particle));
    }
    return particleObservations;
}

std::vector<LandmarkObs> filterPredictedLandmarks(
    const std::vector<LandmarkObs>& observedLandmarks,
    const std::vector<LandmarkObs>& predictedLandmarks)
{
    // Select only map landmarks that correspond to observed ones. Not super
    // efficient, but makes the weight calculation much simpler to have them
    // lined up.
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

double logObservationWeight(const LandmarkObs& predicted, const LandmarkObs& observed,
                         const double stdLandmark[2]) {
    double err_x = predicted.x - observed.x;
    double err_y = predicted.y - observed.y;
    return -(err_x * err_x / stdLandmark[0] / stdLandmark[0]
                + err_y * err_y / stdLandmark[1] / stdLandmark[1])
           / 2;
}

double particleWeight(const Particle& particle, const std::vector<LandmarkObs>& observedLandmarks, const std::vector<LandmarkObs>& predictedLandmarks,  double stdLandmark[2])
{
    // Calculate the weight of a particle given some observations, and some
    // predicted landmarks based on the particles state.
    // Use the log of the joint distribution here. Having a product of small
    // numbers causes issues when there are many particles.
    // Also it is significantly more efficient, since only a single exp call is
    // needed per particle.
    double totalLogWeight = 0.0;
    assert(observedLandmarks.size() == predictedLandmarks.size());
    for(size_t i = 0; i < observedLandmarks.size(); ++i)
    {
        assert(predictedLandmarks[i].id == observedLandmarks[i].id);
        double logWeight = logObservationWeight(predictedLandmarks[i], observedLandmarks[i], stdLandmark);
        totalLogWeight = logWeight + totalLogWeight;
    }
    double normalisation = pow(1 / (2 * M_PI * stdLandmark[0] * stdLandmark[1]),
                               observedLandmarks.size());
    return normalisation * exp(totalLogWeight);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

    std::vector<LandmarkObs> predicted_landmarks;
    predicted_landmarks.reserve(map_landmarks.landmark_list.size());
    for (auto&& landmark : map_landmarks.landmark_list)
    {
        predicted_landmarks.push_back({
                                          landmark.id_i,
                                          landmark.x_f,
                                          landmark.y_f
                                      });
    }

    for (size_t i = 0; i < particles.size(); ++i)
    {
        const auto& particle = particles[i];
        auto particle_observations = predictParticleObservations(observations, particle);
        dataAssociation(predicted_landmarks, particle_observations);
        weights[i] = particleWeight(
            particle,
            filterPredictedLandmarks(particle_observations, predicted_landmarks),
            particle_observations,
            std_landmark);
    }
    double total_weight = accumulate(begin(weights), end(weights), 0.0);
    if (total_weight > 0.001)
    {
        for (size_t i = 0; i < particles.size(); ++i) {
            weights[i] = weights[i] / total_weight;
        }
    }
    else
    {
        // If all of the sampled particles were too far off the mark to generate
        // a consistent distribution, assume that any particle was equally
        // likely. We must ensure that the weights add up to one to make a
        // sensible probability mass function for then resample step.
        for (size_t i = 0; i < particles.size(); ++i)
        {
            weights[i] = 1.0 / particles.size();
        }
    }

    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].weight = weights[i];
    }

}

void ParticleFilter::resample() {
    std::discrete_distribution<size_t> particle_dist(begin(weights), end(weights));
    std::default_random_engine gen;
    std::vector<Particle> new_particles;
    new_particles.reserve(particles.size());
    for(size_t i = 0; i < particles.size(); ++i) {
        new_particles.push_back(particles[particle_dist(gen)]);
    }
    particles = move(new_particles);
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

    particle.associations = associations;
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
