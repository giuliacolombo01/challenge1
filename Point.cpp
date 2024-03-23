//
// Created by Giulia Colombo on 12/03/24.
//

#include "Point.h"
#include <cmath>
#include <iostream>
#include <iomanip>


double Point::distance (const Point & p) const{

    double distance = 0.0;

    for (std::size_t i = 0; i < coordinates.size (); ++i)    {
        distance += std::pow(coordinates[i] - p.get_coordinate(i),2);
    }

    return std::sqrt(distance);

}


std::size_t Point::get_dimension () const{

    return coordinates.size ();

}


double Point::get_coordinate (std::size_t i) const {

    return coordinates[i];

}


void Point::set_coordinate (std::size_t i, double value) {

    while (coordinates.size() <= i){
        coordinates.push_back(0.0);
    }

    coordinates[i] = value;

}


void Point::set_coordinates(const std::vector<double> &coordinates_) {

    coordinates = coordinates_;

}


const std::vector<double> & Point::get_coordinates () const {

    return coordinates;

}


void Point::print() const {

    std::cout << "(";

    for (std::size_t i = 0; i < coordinates.size() - 1; ++i)
        std::cout << std::setprecision(3) << coordinates[i] << ", ";

    std::cout << coordinates[coordinates.size() - 1] << ")" << std::endl;

}