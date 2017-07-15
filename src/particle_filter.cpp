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
#include <limits>

#include "particle_filter.h"

using namespace std;
const double PARTICLE_EPSILON = 0.0001;

inline double multivariate_gaussian(double x, double y, double ux, double uy, double sx, double sy) {
  return exp(-(pow(x - ux,2) / pow(sx,2) + pow(y - uy,2) / pow(sy,2))/2) / (2 * M_PI * sx * sy);
}

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 100;
  
  particles.resize(num_particles);
  weights.resize(num_particles);
  
  default_random_engine generator;
  normal_distribution<double> xdist(x, std[0]);
  normal_distribution<double> ydist(y, std[1]);
  normal_distribution<double> thetadist(theta, std[2]);
  
  for (auto & particle: particles) {
    particle.id = 0;
    particle.x = xdist(generator);
    particle.y = ydist(generator);
    particle.theta = thetadist(generator);
    particle.weight = 1;
  }
  
  for (auto & weight: weights) {
    weight = 1;
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  default_random_engine generator;
  
  for(auto &particle: particles) {
    double theta_m, x_m, y_m;
    
    if (abs(yaw_rate) >= PARTICLE_EPSILON) {
      theta_m = particle.theta + yaw_rate * delta_t;
      x_m = particle.x + velocity * (sin(theta_m) - sin(particle.theta)) / yaw_rate;
      y_m = particle.y + velocity * (cos(particle.theta) - cos(theta_m)) / yaw_rate;
    } else {
      theta_m = particle.theta;
      x_m = particle.x + velocity * delta_t * cos(theta_m);
      y_m = particle.y + velocity * delta_t * sin(theta_m);
    }
    
    normal_distribution<double> x_dist(x_m, std_pos[0]);
    normal_distribution<double> y_dist(y_m, std_pos[1]);
    normal_distribution<double> theta_dist(theta_m, std_pos[2]);
    
    particle.x = x_dist(generator);
    particle.y = y_dist(generator);
    particle.theta = theta_dist(generator);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
  
  for (auto & ob: observations) {
    auto closest = min_element(begin(predicted), end(predicted),
                               [& ob] (const LandmarkObs &a, const LandmarkObs &b)
                               { return dist(a.x, a.y, ob.x, ob.y) < dist(b.x, b.y, ob.x, ob.y); } );
    ob.id = distance(begin(predicted), closest);
  }
  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  
  auto weight = begin(weights);
  for (auto & particle: particles) {
    
    // 1. Transform observations to real world coordinates
    double xp = particle.x;
    double yp = particle.y;
    double thetap = particle.theta;

    vector<LandmarkObs> obs_world = observations;
    for (auto & observation: obs_world) {
      double xo = observation.x;
      double yo = observation.y;
      
      observation.x = xo * cos(thetap) - yo * sin(thetap) + xp;
      observation.y = xo * sin(thetap) + yo * cos(thetap) + yp;
    }
    
    // 2. Find landmarks in range
    vector<LandmarkObs> predicted;
    
    for (auto & map_landmark: map_landmarks.landmark_list) {
      double xm = (double)map_landmark.x_f;
      double ym = (double)map_landmark.y_f;
      
      if (dist(xm, ym, xp, yp) <= sensor_range) {
        LandmarkObs obs;
        obs.x = xm;
        obs.y = ym;
        obs.id = map_landmark.id_i;
        predicted.push_back(obs);
      }
    }
    
    if (predicted.size() != 0) {
      // 3. Associate landmarks with observations using nearest-neighbors
      dataAssociation(predicted, obs_world);
      
      // 4. Calculate final weight using Multivariate-Gaussian probability
      *weight = 1;
      for (auto & observation : obs_world) {
        double xo = observation.x;
        double yo = observation.y;
        double xr = predicted[observation.id].x;
        double yr = predicted[observation.id].y;
        
        (*weight) *= multivariate_gaussian(xo, yo, xr, yr, std_landmark[0], std_landmark[1]);
      }
      
      particle.weight = (*weight);
    }
    weight++;
  }
}

void ParticleFilter::resample() {
  default_random_engine generator;
  discrete_distribution<int> ddist(weights.begin(), weights.end());
  
  vector<Particle> particles_copy = particles;
  for (int i = 0; i < num_particles; i++) {
    particles[i] = particles_copy[ddist(generator)];
  }
}


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  
  //Clear the previous associations
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();
  
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
