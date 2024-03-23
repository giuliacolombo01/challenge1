
# Minimum of a function

The code finds the minimum of a function in different ways.

At first it uses the gradient method starting from an initial guess. It uses as stopping criteria the control on the step length, the control on the residual and it also limits the maximum number of iterations, in which case we have no convergence.
All the necessary parameters are provided by the programmer in the code.

The choice of the step αk to use is critical. By taking μ small and given an initial value α0, we can choose αk by using the exponential decay, the inverse decay and the approximate line search. In particular for the last one the code uses Armijo rule. 
The choice of which method to use is provided by the programmer (by modifing lines 166, 232, 308, 379, 382) and all these possible choices are contained in an enum class.

Then the code uses the momentum method with the same stopping criteria as before. This method updates the value of xk by moving along a direction dk and in particular with this method we cannot apply Armijo rule since the direction dk cannot be guaranteed to be a descent direction, so the code prints out an error in this case.

At last the code uses the Nesterov method with the same stopping criteria as before. Also this method can't be applied with Armijo method.

To implement all these methods the code needs the derivative of the function, or the gradient in dimensions bigger than 1D. It can be calculated by using the exact one or by finite differences and both the methods are provided by the programmer in the code. Which of the two possibility to use is choosen by the user at runtime.