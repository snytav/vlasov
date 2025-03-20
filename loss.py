import torch
from torch.autograd.functional import jacobian
from torch.autograd.functional import hessian
from torch.autograd import grad

def loss_function(x, y,pde,psy_trial,f):
    loss_sum = 0.

    for xi in x:
        for yi in y:
            input_point = torch.Tensor([xi, yi])
            input_point.requires_grad_()

            net_out = pde.forward(input_point)
            net_out_w = grad(outputs=net_out, inputs=pde.fc1.weight, grad_outputs=torch.ones_like(net_out),
                       retain_graph=True,create_graph=True)


            net_out_jacobian = jacobian(pde.forward,input_point,create_graph=True)
            # jac1  = get_jacobian(pde.forward,input_point,2)
            net_out_hessian = hessian(pde.forward,input_point,create_graph=True)
            psy_t = psy_trial(input_point, net_out)

            inputs = (input_point, net_out)
            psy_t_jacobian = jacobian(psy_trial, inputs,create_graph=True)[0]
            psy_t_hessian  = hessian(psy_trial,inputs,create_graph=True)
            psy_t_hessian = psy_t_hessian[0][0]
            # acobian(jacobian(psy_trial))(input_point, net_out

            gradient_of_trial_d2x = psy_t_hessian[0][0]
            gradient_of_trial_d2y = psy_t_hessian[1][1]

            # D_gradient_of_trial_d2x_D_W0 = grad(outputs=gradient_of_trial_d2x, inputs=pde.fc1.weight, grad_outputs=torch.ones_like(gradient_of_trial_d2x), retain_graph=True)
            # D_gradient_of_trial_d2y_D_W0 = grad(outputs=gradient_of_trial_d2y, inputs=pde.fc1.weight, grad_outputs=torch.ones_like(gradient_of_trial_d2y), retain_graph=True)
            # D_func_D_W0 = grad(outputs=func,iputs=pde.fc1.weight,grad_outputs=torch.ones_like(func))
            func = f(input_point)
            func_t = torch.Tensor([func])
            func_t.requires_grad_()

            err_sqr = ((gradient_of_trial_d2x + gradient_of_trial_d2y) - func_t) ** 2
            # D_err_sqr_D_W0 = 2*((gradient_of_trial_d2x + gradient_of_trial_d2y) - func)*(
            #                     (D_gradient_of_trial_d2x_D_W0 + D_gradient_of_trial_d2y_D_W0) -D_func_D_W0
            #                     )

            loss_sum += err_sqr
            qq = 0

    return loss_sum
