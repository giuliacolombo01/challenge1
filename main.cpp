#include <iostream>
#include <cmath>
#include <functional>

#include "Point.h"

//Create a structure containing all the parameters needed
struct parameters {
    Point x0;
    double tolerance_r;
    double tolerance_s;
    std::function<double(Point)> funct;  //This is the function to minimize
    std::function<std::vector<double>(Point, parameters&)> grad;  //This is the gradient of the function
    double alpha;
    int max_iter;
};

//Create an enum class containing all the possible strategies to update alpha
enum class type_alpha {exponential, inverse, Armijo};

//Insert the function
std::function<double(Point)> funct = [] (Point x) -> double {

    double result;

    result = x.get_coordinate(0) * x.get_coordinate(1) + 4 * std::pow(x.get_coordinate(0), 4) + std::pow(x.get_coordinate(1), 2) + 3 * x.get_coordinate(0);
    return result;
};

//Insert the function that evaluates the exact gradient of the function component wise
std::function<std::vector<double>(Point, parameters&)> grad_exact = [] (Point x, parameters& param) -> std::vector<double> {

    std::vector<double> result(x.get_dimension());

    result[0] = x.get_coordinate(1) + 16 * std::pow(x.get_coordinate(0), 3) + 3;
    result[1] = x.get_coordinate(0) + 2 * x.get_coordinate(1);
    return result;
};

//Insert the function that evaluates the gradient of the function by finite differences
std::function<std::vector<double>(Point, parameters&)> grad_finite = [] (Point x, parameters& param) -> std::vector<double> {

    //Insert a value of the step
    double h = 0.01;

    std::vector<double> result(x.get_dimension());

    //Create two points (they will be x+h and x-h on each component)
    Point xph(std::vector<double> (x.get_dimension(), 0.));
    Point xmh(std::vector<double> (x.get_dimension(), 0.));

    for (std::size_t i = 0; i < x.get_dimension(); ++i) {

        //Set the points equal to x
        xph = x;
        xmh = x;

        //Add h on the component i
        xph.set_coordinate(i, x.get_coordinate(i) + h);
        xmh.set_coordinate(i, x.get_coordinate(i) - h);

        //Evaluate the result by centered finite difference
        result[i] = (param.funct(xph) - param.funct(xmh)) / (2 * h);
    }

    return result;
};

//This function evaluates the norm of a given vector
double norm(std::vector<double> v) {

    //Initialize the result
    double result = 0;

    for (std::size_t i = 0; i < v.size(); ++i) {
        //Add to the previous result a new component squared
        result += std::pow(v[i], 2);
    }

    //Return the square root of the sum to have the norm
    return std::sqrt(result);
}

//This function updates the value of alpha at each iteration
template<type_alpha update_alpha>
double evaluate_alpha(parameters& param, int iter){

    //Initialize the value of a new alpha equal to the one given
    double alpha1 = param.alpha;

    //Initialize the parameters needed in the algorithm
    double mu = 0.2;
    double sigma = 0.3;

    //Verify which strategy is required
    if constexpr (update_alpha == type_alpha::exponential) {

        alpha1 = alpha1 * std::exp(-mu * iter);

    } else if constexpr (update_alpha == type_alpha::inverse) {

        alpha1 = alpha1 / (1 + mu * iter);

    } else if constexpr (update_alpha == type_alpha::Armijo) {

        //Create a new point usefull in the algorithm
        Point x(std::vector<double> (param.x0.get_dimension(), 0.));

        for (std::size_t i = 0; i < param.x0.get_dimension(); ++i) {
            x.set_coordinate(i, param.x0.get_coordinate(i) - alpha1 * param.grad(param.x0, param)[i]);
        }

        //Keep on updating alpha since the condition is verified
        while (param.funct(param.x0) - param.funct(x) < sigma * alpha1 * norm(param.grad(param.x0, param))) {
            alpha1 = alpha1 / 2;

            //Update the coordinates of the created point
            for (std::size_t i = 0; i < param.x0.get_dimension(); ++i) {
                x.set_coordinate(i, param.x0.get_coordinate(i) - alpha1 * std::pow(param.grad(param.x0, param)[i], 2));
            }
        }
    } else {

        //If the strategy is not recognized the code prints out a error
        std::cout << "Error: invalid strategy!" << std::endl;
    }

    return alpha1;
}

//This function finds the minimum of the function by using gradient method
Point minimum_grad(parameters& param) {

    //Create a point which will be the minimum (which will be x(k+1))
    Point x1(std::vector<double> (param.x0.get_dimension(), 0.));

    //Create a point which stores the old point (x(k))
    Point x0 = param.x0;

    //Initialize the parameters that I want to control equal to the tolerance
    double err = param.tolerance_s;
    double grad = param.tolerance_r;

    //Initialize the number of iterations to 0
    int iter = 0;

    //Initialize the step alpha
    double alpha_k = param.alpha;

    while (err >= param.tolerance_s && grad >= param.tolerance_r && iter < param.max_iter) {

        //Evaluate and store the derivative in a new variable so that it is not evaluated at each repetition of the for cicle
        std::vector<double> deriv = param.grad(x0, param);

        //Evaluate the new point which could possibly be the minimum
        for (std::size_t i = 0; i < x0.get_dimension(); ++i) {
            x1.set_coordinate(i, x0.get_coordinate(i) - alpha_k * deriv[i]);
        }

        //Update the values of the parameters that I want to control
        err = x0.distance(x1);
        grad = norm(param.grad(x0, param));
        ++iter;

        //Evaluate the new value of alpha (change the template variable of the function to have a different strategy to update it)
        alpha_k = evaluate_alpha<type_alpha::inverse>(param, iter);

        //Update the value of old point (x0) equal to the new one (x1)
        x0 = x1;
    }

    //Print the minimum
    std::cout << "The point of minimum with gradient method is: " << std::endl;
    x1.print();

    //Print the value of the function in the minimum
    std::cout << "The value of this minimum is: " << param.funct(x1) << std::endl;
    std::cout << std::endl;

    return x1;
}

//This function finds the minimum of the function by using momentum method
template<type_alpha update_alpha>
Point minimum_momentum(parameters& param) {

//Create a point which stores the old point (x(k))
    Point x0 = param.x0;

    if constexpr (update_alpha == type_alpha::Armijo) {

        //Nesterov method can't be applied in this case
        std::cout << "Warning: with Armijo method the direction dk cannot be guaranteed to be a descent direction so the momentum method doesn't apply" << std::endl;
        std::cout << std::endl;

    } else if constexpr (update_alpha == type_alpha::exponential || update_alpha == type_alpha::inverse) {

        //Initialize the step alpha
        double alpha_k = param.alpha;

        //Initialize the constant parameter
        double nu = 0.9;

        //Create a point which will be the minimum (which will be x(k+1))
        Point x1(std::vector<double> (param.x0.get_dimension(), 0.));

        //Initialize the point and also y
        for (std::size_t i = 0; i < param.x0.get_dimension(); ++i) {
            x1.set_coordinate(i, x0.get_coordinate(i) - alpha_k * param.grad(x0, param)[i]);
        }

        //Initialize the parameters that I want to control equal to the tolerance
        double err = param.tolerance_s;
        double grad = param.tolerance_r;

        //Initialize the number of iterations to 0
        int iter = 0;

        while (err >= param.tolerance_s && grad >= param.tolerance_r && iter < param.max_iter) {

            //Evaluate the new point which could possibly be the minimum
            for (std::size_t i = 0; i < param.x0.get_dimension(); ++i) {
                x1.set_coordinate(i, x1.get_coordinate(i) - alpha_k * param.grad(x1, param)[i] + nu * (x1.get_coordinate(i) - x0.get_coordinate(i)));
            }

            //Update the values of the parameters that I want to control
            err = x0.distance(x1);
            grad = norm(param.grad(x0, param));
            ++iter;

            //Evaluate the new value of alpha (change the template variable of the function to have a different strategy to update it)
            alpha_k = evaluate_alpha<type_alpha::inverse>(param, iter);

            //Update the value of old point (x0) equal to the new one (x1)
            x0 = x1;
        }

        //Print the minimum
        std::cout << "The point of minimum with momentum method is: " << std::endl;
        x1.print();

        //Print the value of the function in the minimum
        std::cout << "The value of this minimum is: " << param.funct(x1) << std::endl;
        std::cout << std::endl;

    } else {

        //If the strategy is not recognized the code prints out a error
        std::cout << "Error: invalid strategy!" << std::endl;
        std::cout << std::endl;
    }

    return x0;

}

//This function finds the minimum of the function by using Nesterov method
template<type_alpha update_alpha>
Point minimum_nesterov(parameters& param) {

    //Create a point which stores the old point (x(k))
    Point x0 = param.x0;

    if constexpr (update_alpha == type_alpha::Armijo) {

        //Nesterov method can't be applied in this case
        std::cout << "Warning: with Armijo method also Nesterov method doesn't apply" << std::endl;
        std::cout << std::endl;

    } else if constexpr (update_alpha == type_alpha::exponential || update_alpha == type_alpha::inverse) {

        //Initialize the step alpha
        double alpha_k = param.alpha;

        //Initialize the constant parameter
        double nu = 0.9;

        //Create a point which will be the minimum (which will be x(k+1)) and y
        Point x1(std::vector<double> (param.x0.get_dimension(), 0.));
        Point y(std::vector<double> (param.x0.get_dimension(), 0.));

        //Initialize the point and also y
        for (std::size_t i = 0; i < param.x0.get_dimension(); ++i) {
            x1.set_coordinate(i, x0.get_coordinate(i) - alpha_k * param.grad(x0, param)[i]);
            y.set_coordinate(i, x1.get_coordinate(i) + nu * (x1.get_coordinate(i) - x0.get_coordinate(i)));
        }

        //Initialize the parameters that I want to control equal to the tolerance
        double err = param.tolerance_s;
        double grad = param.tolerance_r;

        //Initialize the number of iterations to 0
        int iter = 0;

        while (err >= param.tolerance_s && grad >= param.tolerance_r && iter < param.max_iter) {

            //Evaluate the new point which could possibly be the minimum
            for (std::size_t i = 0; i < param.x0.get_dimension(); ++i) {
                x1.set_coordinate(i, y.get_coordinate(i) - alpha_k * param.grad(y, param)[i]);
            }

            //Update the values of the parameters that I want to control
            err = x0.distance(x1);
            grad = norm(param.grad(x0, param));
            ++iter;

            //Evaluate the new value of alpha (change the template variable of the function to have a different strategy to update it)
            alpha_k = evaluate_alpha<type_alpha::inverse>(param, iter);

            //Update y
            for (std::size_t i = 0; i < param.x0.get_dimension(); ++i) {
                y.set_coordinate(i, x1.get_coordinate(i) + nu * (x1.get_coordinate(i) - x0.get_coordinate(i)));
            }

            //Update the value of old point (x0) equal to the new one (x1)
            x0 = x1;
        }

        //Print the minimum
        std::cout << "The point of minimum with Nesterov method is: " << std::endl;
        x1.print();

        //Print the value of the function in the minimum
        std::cout << "The value of this minimum is: " << param.funct(x1) << std::endl;
        std::cout << std::endl;

    } else {

        //If the strategy is not recognized the code prints out a error
        std::cout << "Error: invalid strategy!" << std::endl;
        std::cout << std::endl;
    }

    return x0;

}

int main() {

    parameters param;

    //Insert the initial point here below
    param.x0 = Point(std::vector<double> (2, 0.));

    //Insert the tolerance_r here below
    param.tolerance_r = 1e-6;

    //Insert the tolerance_s here below
    param.tolerance_s = 1e-6;

    //Initialize the function that you want to minimize
    param.funct = funct;

    int type_grad = 0;

    //Choose the strategy to evaluate the derivative
    std::cout << "Insert the strategy to evaluate the derivative (1 for the exact derivative, 2 for finite differences)" << std::endl;
    std::cin >> type_grad;

    //Initialize the gradient of the function that you want to minimize
    if (type_grad == 1) {
        param.grad = grad_exact;
    } else if (type_grad == 2) {
        param.grad = grad_finite;
    } else {
        std::cout << "Error: invalid strategy!" << std::endl;
    };

    //Insert the initial step alpha_0 here below
    param.alpha = 0.01;

    //Insert the max number of iterations allowed here below
    param.max_iter = 100;

    //Find the minimum with gradient method
    Point min_grad = minimum_grad(param);

    //Find the minimum with momentum method
    Point min_mom = minimum_momentum<type_alpha::inverse>(param);

    //Find the minimum with Nesterov method
    Point min_nest = minimum_nesterov<type_alpha::inverse>(param);

    return 0;
}