import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import sympy as sp
from matplotlib import cm

x, y, t = sp.symbols("x,y,t")


class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""

        self.N = N
        self.L = 1
        self.h = self.L / N
        x_l = np.linspace(0, self.L, N + 1)
        y_l = np.linspace(0, self.L, N + 1)
        self.xij, self.yij = np.meshgrid(x_l, y_l, indexing="ij", sparse=sparse)

    def D2(self, N):
        """Return second order differentiation matrix"""

        res = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1), "lil")
        res[0, :4] = 2, -5, 4, -1
        res[-1, -4:] = -1, 4, -5, 2

        return res.tocsr()

    @property
    def w(self):
        """Return the dispersion coefficient"""

        return self.c * np.sqrt(np.pi * np.pi * (self.mx * self.mx + self.my * self.my))

    def ue(self, mx, my):
        """Return the exact standing wave"""

        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """

        # Use ue to set u0
        self.unm1 = sp.lambdify((x, y, t), self.ue(mx, my))(self.xij, self.yij, 0.0)


        D = self.D2(N) / self.h**2

        # Use u0 to set u1
        self.un = self.unm1 + 0.5 * (self.c * self.dt) ** 2 * (
            D @ self.unm1 + self.unm1 @ D.T
        )


    @property
    def dt(self):
        """Return the time step"""

        return self.cfl*self.h/self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """

        ue = sp.lambdify((x, y, t), self.ue(self.mx, self.my))(self.xij, self.yij, t0)

        err = np.sqrt(self.h * self.h * np.sum((u - ue) ** 2))

        return err

    def apply_bcs(self):
        self.unp1[0] = 0
        self.unp1[-1] = 0
        self.unp1[:, 0] = 0
        self.unp1[:, -1] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """

        self.cfl = cfl
        self.c = c
        self.mx = mx
        self.my = my

        self.create_mesh(N)

        self.unp1, self.un, self.unm1 = np.zeros((3, N + 1, N + 1))

        self.initialize(N, mx, my)

        D = self.D2(N) / self.h**2

        errors = []
        for n in range(1, Nt):
            self.unp1 = (
                2 * self.un
                - self.unm1
                + (c * self.dt) ** 2 * (D @ self.un + self.un @ D.T)
            )

            self.apply_bcs()

            self.un, self.unm1 = self.unp1, self.un

            # n+1 since un == unp1
            errors.append(self.l2_error(self.un, (n+1)*self.dt))

        return (self.h, errors)

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [
            np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i])
            for i in range(1, m + 1, 1)
        ]
        print(E)
        print(r)
        return r, np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):

    def D2(self, N):

        res = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1), "lil")
        res[0, :4] = -2, 2, 0, 0
        res[-1, -4:] = 0, 0, 2, -2

        return res.tocsr()

    def ue(self, mx, my):
        return sp.cos(mx * sp.pi * x) * sp.cos(my * sp.pi * y) * sp.cos(self.w * t)

    def apply_bcs(self):
        return


def test_convergence_wave2d():
    sol = Wave2D()
    r, _, _ = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, _, _ = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1] - 2) < 0.05


def test_exact_wave2d():
    raise NotImplementedError
