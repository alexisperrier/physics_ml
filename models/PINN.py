from typing import Any, Callable, Dict
import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn, autograd

class OdePINN(pl.LightningModule):
    def __init__(
            self,
            input_size: int,     # 1 for t, n for R^n
            hidden_size: int,
            output_size: int,    # u in R^output_size
            n_layer: int,
            lr: float = 1e-3,
            seed: int = 42,
            t_max: int = 50,
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.save_hyperparameters()

        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(n_layer - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, output_size), nn.Softplus()]      # enforce positive valued output

        self.net = nn.Sequential(*layers)
        self.mse = nn.MSELoss()
        self.n = input_size
        self.m = output_size        # state dimension
        self.lr= lr
        self.t_max = t_max

    def forward(self,
                t: torch.Tensor) -> torch.Tensor:
        # t: (N, input_size)
        # returns u: (N, output_size)
        t_norm = t / self.t_max   # normalized t, [0,T] --> [0,1]
        return self.net(t_norm)
    
    def u(self, 
          t: torch.Tensor) -> torch.Tensor:
        return self.forward(t)
    
    def residual(self, 
                 t: torch.Tensor) -> torch.Tensor:
        """
        Residual of ODE system at time t
        """
        t = t.clone().detach().requires_grad_(True)      # (N, input_size)
        u_out = self.u(t)                                # (N, m)

        # compute du_j/dt for each component j = 0,...,m-1
        du_dt_components = []
        for j in range(self.m):
            u_j = u_out[:, j:j+1]                       # (N, 1)
            du_j_dt = autograd.grad(                    # (N, 1)
                outputs=u_j,
                inputs=t,
                grad_outputs=torch.ones_like(u_j),
                create_graph=True,
                retain_graph=True,
            )[0]
            du_dt_components.append(du_j_dt)

        du_dt = torch.cat(du_dt_components, dim=1)      # (N, m)

        f = self.ode_rhs(t, u_out)                      # (N, m)

        residual = du_dt - f                            # (N, m)

        return residual

    def compute_loss(self,
                     batch: Dict[str, torch.Tensor],
                     ) -> Dict[str, torch.Tensor]:
        
        t = batch["t"].to(self.device).view(-1, self.n)          # (N,1)
        t0 = batch["t0"].to(self.device).view(-1, 1)             # (N,1)
        u_res = batch["u_res"].to(self.device).view(-1, self.m)  # 0 array in R^m (N,m)
        u0 = batch["u0"].to(self.device).view(-1, self.m)        # (N,m)

        # Residual loss
        loss_res = self.mse(self.residual(t=t), u_res)

        # Initial condition loss
        loss_ic = self.mse(self.u(t0), u0)

        loss_total = 100.0 * loss_res + 50.0 * loss_ic

        return {
            "loss_total": loss_total,
            "loss_res": loss_res,
            "loss_ic": loss_ic,
        }

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        losses = self.compute_loss(batch)
        self.log("train/loss", losses["loss_total"], prog_bar=True)
        self.log("train/loss_res", losses["loss_res"], prog_bar=False)
        self.log("train/loss_ic", losses["loss_ic"], prog_bar=False)
        return losses["loss_total"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def ode_rhs(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        f(t, u) in du/dt = f(t, u)
        Expected shape: f(t,u) -> (N, output_size)
        """
        raise NotImplementedError
    

class LotkaVolterraOdePINN(OdePINN):
    """
    PINN is u(t) solution for single predetermined trajectory (set of system parameters) 
    in du/dt = f(t, u)
    """
    def __init__(
            self,
            hidden_size: int,
            n_layer: int,
            lr: float = 1e-3,
            alpha: float = 1.5,
            beta: float = 1.0,
            gamma: float = 3.0,
            delta: float = 1.0,
            t_max: int = 50
            ):
        super().__init__(input_size=1, hidden_size=hidden_size, output_size=2, n_layer=n_layer, lr=lr, t_max=t_max)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def ode_rhs(self,
                t: torch.Tensor,
                u: torch.Tensor,) -> torch.Tensor:
        """
        Lotka–Volterra RHS:
            u = [x, y]
            x' = alpha * x - beta * x * y
            y' = delta * x * y - gamma * y

        Args:
            t: (N,1) time tensor (not used directly in LV
            u: (N,2) tensor with u[:,0]=x, u[:,1]=y

        Returns:
            f(t,u): (N,2) tensor with [x', y']
        """
        x = u[:, 0:1]
        y = u[:, 1:2]

        x_dot = self.alpha * x  - self.beta * x * y
        y_dot = self.delta * x * y -self.gamma * y

        f = torch.cat([x_dot, y_dot], dim=1)
        return f
        
class ParameterAgnosticOdePINN(OdePINN):
    """
    Parameter vectors are treated as conditioning variables; the network learns u(t, θ) that generalizes across parameter sets.
    """
    def __init__(
            self,
            input_size: int,     # 1 for t, n for R^n
            theta_dim: int,
            hidden_size: int,
            output_size: int,    # u in R^output_size
            n_layer: int,
            lr: float = 1e-3,
            seed: int = 42,
            t_max: int = 50,
            loss_weights: Dict[str, float] | None = None,
    ):
        super().__init__(input_size=input_size, hidden_size=hidden_size, output_size=output_size, n_layer=n_layer, lr=lr, seed=seed, t_max=t_max)
        torch.manual_seed(seed)
        self.save_hyperparameters()

        layers = [nn.Linear(input_size + theta_dim, hidden_size), nn.ReLU()]
        for _ in range(n_layer - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, output_size), nn.Softplus()]      # enforce positive valued output

        self.net = nn.Sequential(*layers)
        self.mse = nn.MSELoss()
        self.n = input_size
        self.p = theta_dim          # parameter dimension
        self.m = output_size        # state dimension
        self.lr = float(lr)
        self.t_max = t_max
        lw = loss_weights or {}
        self.loss_w_res = lw.get("residual", 1.0)
        self.loss_w_ic = lw.get("ic", 1.0)
        self.loss_w_data = lw.get("data", 1.0)

    def forward(self, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        t_norm = t / self.t_max
        t_theta = torch.cat([t_norm, theta], dim=1)
        return self.net(t_theta)
    
    def u(self, t: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        return self.forward(t, theta)
    
    def compute_loss(self,
                     batch: Dict[str, torch.Tensor],
                     ) -> Dict[str, torch.Tensor]:
        
        t = batch["t"].to(self.device).view(-1, self.n)          # (N,1)
        t0 = batch["t0"].to(self.device).view(-1, 1)             # (N,1)
        u_res = batch["u_res"].to(self.device).view(-1, self.m)  # 0 array in R^m (N,m)
        u0 = batch["u0"].to(self.device).view(-1, self.m)        # (N,m)
        theta = batch["theta"].view(-1, self.p)
        t_regression = batch["t_regression"].to(self.device).view(-1, self.n)
        u_regression = batch["u_regression"].to(self.device).view(-1, self.m)

        # Residual loss
        loss_res = self.mse(self.residual(t, theta), u_res)

        # Initial condition loss
        loss_ic = self.mse(self.u(t0, theta), u0)

        # Regression loss
        loss_data = self.mse(
            self.u(t_regression, theta),
            u_regression
        )

        loss_total = (
            self.loss_w_res * loss_res
            + self.loss_w_ic * loss_ic
            + self.loss_w_data * loss_data
        )
        return {
            "loss_total": loss_total,
            "loss_res": loss_res,
            "loss_ic": loss_ic,
            "loss_data": loss_data,
        }

    def residual(self, 
                 t: torch.Tensor,
                 theta: torch.Tensor,
                 ) -> torch.Tensor:
        """
        Residual of ODE system at time t
        """
        t = t.clone().detach().requires_grad_(True)     # (N, input_size)
        theta = theta.clone().detach()
        u_out = self.u(t, theta)                        # (N, m)

        # compute du_j/dt for each component j = 0,...,m-1
        du_dt_components = []
        for j in range(self.m):
            u_j = u_out[:, j:j+1]                       # (N, 1)
            du_j_dt = autograd.grad(                    # (N, 1)
                outputs=u_j,
                inputs=t,
                grad_outputs=torch.ones_like(u_j),
                create_graph=True,
                retain_graph=True,
            )[0]
            du_dt_components.append(du_j_dt)

        du_dt = torch.cat(du_dt_components, dim=1)      # (N, m)

        f = self.ode_rhs(t, u_out, theta)

        residual = du_dt - f                            # (N, m)

        return residual

    def ode_rhs(self, t: torch.Tensor, theta: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        f(t, u) in du/dt = f(t, u)
        Expected shape: f(t,u) -> (N, output_size)
        theta are system parameters
        """
        raise NotImplementedError
    
class LVOdePINN(ParameterAgnosticOdePINN):
    def ode_rhs(self, 
                t: torch.Tensor, 
                u: torch.Tensor, 
                theta: torch.Tensor) -> torch.Tensor:
        
        alpha, beta, gamma, delta = torch.split(theta, 1, dim=1)
        x, y = u[:, :1], u[:, 1:]
        x_dot = alpha * x - beta * x * y
        y_dot = delta * x * y - gamma * y

        return torch.cat([x_dot, y_dot], dim=1)
    
class ParameterICAgnosticOdePINN(ParameterAgnosticOdePINN):
    def __init__(
            self,
            input_size: int,     # 1 for t, n for R^n
            theta_dim: int,
            hidden_size: int,
            output_size: int,    # u in R^output_size
            n_layer: int,
            lr: float = 1e-3,
            seed: int = 42,
            t_max: int = 50,
            loss_weights: Dict[str, float] | None = None,
    ):
        super().__init__(input_size=input_size, theta_dim=theta_dim, hidden_size=hidden_size, output_size=output_size, n_layer=n_layer, lr=lr, seed=seed, t_max=t_max)
        torch.manual_seed(seed)
        self.save_hyperparameters()

        layers = [nn.Linear(input_size + theta_dim + output_size, hidden_size), nn.ReLU()]
        for _ in range(n_layer - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, output_size), nn.Softplus()]      # enforce positive valued output

        self.net = nn.Sequential(*layers)
        self.mse = nn.MSELoss()
        self.n = input_size
        self.p = theta_dim          # parameter dimension
        self.m = output_size        # state dimension
        self.lr = float(lr)
        self.t_max = t_max
        lw = loss_weights or {}
        self.loss_w_res = lw.get("residual", 1.0)
        self.loss_w_ic = lw.get("ic", 1.0)
        self.loss_w_data = lw.get("data", 1.0)

    def forward(self, t: torch.Tensor, theta: torch.Tensor, u0: torch.Tensor) -> torch.Tensor:
        t_norm = t / self.t_max
        t_theta_u0 = torch.cat([t_norm, theta, u0], dim=1)
        return self.net(t_theta_u0)
    
    def u(self, t: torch.Tensor, theta: torch.Tensor, u0: torch.Tensor) -> torch.Tensor:
        return self.forward(t, theta, u0)

    def compute_loss(self,
                     batch: Dict[str, torch.Tensor],
                     ) -> Dict[str, torch.Tensor]:
        
        t = batch["t"].to(self.device).view(-1, self.n)          # (N,1)
        t0 = batch["t0"].to(self.device).view(-1, 1)             # (N,1)
        u_res = batch["u_res"].to(self.device).view(-1, self.m)  # 0 array in R^m (N,m)
        u0 = batch["u0"].to(self.device).view(-1, self.m)        # (N,m)
        theta = batch["theta"].view(-1, self.p)
        t_regression = batch["t_regression"].to(self.device).view(-1, self.n)
        u_regression = batch["u_regression"].to(self.device).view(-1, self.m)

        # Residual loss
        loss_res = self.mse(self.residual(t, theta, u0), u_res)

        # Initial condition loss
        loss_ic = self.mse(self.u(t0, theta, u0), u0)

        # Regression loss
        loss_data = self.mse(
            self.u(t_regression, theta, u0),
            u_regression
        )

        loss_total = (
            self.loss_w_res * loss_res
            + self.loss_w_ic * loss_ic
            + self.loss_w_data * loss_data
        )
        return {
            "loss_total": loss_total,
            "loss_res": loss_res,
            "loss_ic": loss_ic,
            "loss_data": loss_data,
        }

    def residual(self, 
                 t: torch.Tensor,
                 theta: torch.Tensor,
                 u0: torch.Tensor,
                 ) -> torch.Tensor:
        """
        Residual of ODE system at time t
        """
        t = t.clone().detach().requires_grad_(True)     # (N, input_size)
        theta = theta.clone().detach()
        u_out = self.u(t, theta, u0)                        # (N, m)

        # compute du_j/dt for each component j = 0,...,m-1
        du_dt_components = []
        for j in range(self.m):
            u_j = u_out[:, j:j+1]                       # (N, 1)
            du_j_dt = autograd.grad(                    # (N, 1)
                outputs=u_j,
                inputs=t,
                grad_outputs=torch.ones_like(u_j),
                create_graph=True,
                retain_graph=True,
            )[0]
            du_dt_components.append(du_j_dt)

        du_dt = torch.cat(du_dt_components, dim=1)      # (N, m)

        f = self.ode_rhs(t, u_out, theta)

        residual = du_dt - f                            # (N, m)

        return residual

class LVOdePICPINN(ParameterICAgnosticOdePINN):
    def ode_rhs(self, 
                t: torch.Tensor, 
                u: torch.Tensor, 
                theta: torch.Tensor) -> torch.Tensor:
        
        alpha, beta, gamma, delta = torch.split(theta, 1, dim=1)
        x, y = u[:, :1], u[:, 1:]
        x_dot = alpha * x - beta * x * y
        y_dot = delta * x * y - gamma * y

        return torch.cat([x_dot, y_dot], dim=1)
    