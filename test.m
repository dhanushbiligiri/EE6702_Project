clear; clc;

%% ============================================================
% 0. Python environment
%% ============================================================

pe = pyenv;
if pe.Status == "NotLoaded"
    pyenv('Version', '/home/dnarsipu/anaconda3/envs/EE6702/bin/python');
end

pe = pyenv;
fprintf('Using Python:\n%s\n\n', string(pe.Executable));

%% ============================================================
% 1. Add current folder to Python path
%% ============================================================

folder = pwd;
if count(py.sys.path, folder) == 0
    insert(py.sys.path, int32(0), folder);
end
py.importlib.invalidate_caches();

%% ============================================================
% 2. Import Python MuJoCo wrapper
%% ============================================================

mod = py.importlib.import_module('humanoid_dynamics');

%% ============================================================
% 3. Load XML model
%% ============================================================

xml_path = '/home/dnarsipu/Documents/Personal/EE6702_Project/humanoid.xml';
dyn = mod.HumanoidDynamics(xml_path);

nq = double(dyn.nq);
nv = double(dyn.nv);
nu = double(dyn.nu);

fprintf('Model loaded successfully.\n');
fprintf('nq = %d, nv = %d, nu = %d\n', nq, nv, nu);
fprintf('State dimension nx = nq + nv = %d\n\n', nq + nv);

%% ============================================================
% 4. Build initial state x0 = [qpos; qvel]
%% ============================================================

x0 = zeros(nq + nv, 1);

% For floating-base humanoid models:
% qpos(1:3)   -> root position
% qpos(4:7)   -> root quaternion [w x y z]
%
% These are model-dependent assumptions, but common for MuJoCo humanoids.

if nq >= 3
    x0(3) = 1.2;    % initial root height guess
end

if nq >= 7
    x0(4) = 1.0;    % valid unit quaternion
    x0(5) = 0.0;
    x0(6) = 0.0;
    x0(7) = 0.0;
end

u0 = zeros(nu, 1);

fprintf('Initial state and control created.\n');
fprintf('||x0|| = %.6f\n', norm(x0));
fprintf('||u0|| = %.6f\n\n', norm(u0));

%% ============================================================
% 5. Evaluate continuous-time dynamics xdot = f(x,u,t)
%% ============================================================

% Current implementation is time-invariant, so t is implicit.
xdot0 = humanoid_f(x0, u0, dyn);

fprintf('Dynamics evaluated.\n');
fprintf('size(xdot0) = [%d x %d]\n', size(xdot0,1), size(xdot0,2));
fprintf('||xdot0|| = %.6f\n\n', norm(xdot0));

%% ============================================================
% 6. Split derivative into [qdot; vdot]
%% ============================================================

qdot0 = xdot0(1:nq);
vdot0 = xdot0(nq+1:end);

fprintf('Top part: qdot = derivative of qpos\n');
fprintf('Bottom part: vdot = qacc = derivative of qvel\n\n');

fprintf('||qdot0|| = %.6f\n', norm(qdot0));
fprintf('||vdot0|| = %.6f\n\n', norm(vdot0));

disp('First 10 entries of qdot0:');
disp(qdot0(1:min(10,length(qdot0))));

disp('Last 10 entries of vdot0:');
disp(vdot0(max(1,end-9):end));

%% ============================================================
% 7. Perturb control and check whether dynamics change
%% ============================================================

u1 = zeros(nu,1);
u1(1) = 0.5;

xdot1 = humanoid_f(x0, u1, dyn);
delta_u_response = norm(xdot1 - xdot0);

fprintf('\nControl perturbation test:\n');
fprintf('Set u1(1) = 0.5\n');
fprintf('||f(x0,u1) - f(x0,u0)|| = %.6f\n\n', delta_u_response);

%% ============================================================
% 8. Perturb state and check whether dynamics change
%% ============================================================

x1 = x0;
x1(1) = x1(1) + 1e-3;

xdot_state_pert = humanoid_f(x1, u0, dyn);
delta_x_response = norm(xdot_state_pert - xdot0);

fprintf('State perturbation test:\n');
fprintf('Perturbed x0(1) by 1e-3\n');
fprintf('||f(x1,u0) - f(x0,u0)|| = %.6f\n\n', delta_x_response);

%% ============================================================
% 9. One explicit Euler step
%% ============================================================

dt = 0.01;
x_next = x0 + dt * xdot0;

fprintf('One-step Euler simulation:\n');
fprintf('dt = %.4f\n', dt);
fprintf('||x_next|| = %.6f\n\n', norm(x_next));

%% ============================================================
% 10. Summary prints for teammate reporting
%% ============================================================

fprintf('==================== SUMMARY ====================\n');
fprintf('XML loaded successfully.\n');
fprintf('Continuous-time dynamics evaluated in the form:\n');
fprintf('    xdot = f(x,u,t)\n');
fprintf('with state defined as:\n');
fprintf('    x = [qpos; qvel]\n');
fprintf('Dimensions:\n');
fprintf('    nq = %d, nv = %d, nu = %d, nx = %d\n', nq, nv, nu, nq+nv);
fprintf('Returned derivative size:\n');
fprintf('    xdot in R^{%d}\n', nq+nv);
fprintf('Control response check:\n');
fprintf('    ||f(x0,u1)-f(x0,u0)|| = %.6f\n', delta_u_response);
fprintf('State response check:\n');
fprintf('    ||f(x1,u0)-f(x0,u0)|| = %.6f\n', delta_x_response);
fprintf('=================================================\n');