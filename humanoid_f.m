function xdot = humanoid_f(x, u, dyn)
    np = py.importlib.import_module('numpy');

    x_py = np.array(x(:).', pyargs('dtype', 'float64'));
    u_py = np.array(u(:).', pyargs('dtype', 'float64'));

    xdot_py = dyn.f(x_py, u_py);

    xdot = double(xdot_py);
    xdot = xdot(:);
end