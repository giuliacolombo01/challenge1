//
// Created by Giulia Colombo on 12/03/24.
//

#ifndef CHALLENGE1_POINT_H
#define CHALLENGE1_POINT_H

#include<vector>
#include <iostream>

class Point {

private:
    std::vector<double> coordinates;

public:

    // Constructor
    Point () = default;
    Point (std::vector<double> const & coordinates_): coordinates (coordinates_) {};

    // Compute the distance from a point p
    double distance (const Point & p) const;

    // Get the dimension of the input points
    std::size_t get_dimension () const;

    // Get the value of the i-th coordinate
    double get_coordinate (std::size_t i) const;

    // Set the value of i-th coordinate
    void set_coordinate (std::size_t i, double value);

    // Getter and setters
    void set_coordinates (const std::vector<double> & coordinates_);
    const std::vector<double> & get_coordinates () const;

    // Print the coordinate of the point
    void print () const;

};

#endif //CHALLENGE1_POINT_H
