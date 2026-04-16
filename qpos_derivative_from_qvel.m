function qpos_dot = qpos_derivative_from_qvel(model, qpos, qvel)
% Convert generalized velocity qvel into a qpos-type derivative.
%
% Important:
% For Humanoid, the root orientation is a quaternion, so nq > nv.
% You cannot use qpos_dot = qvel directly.
%
% This function uses MuJoCo's quaternion-aware position integration:
%   qpos_next = integratePos(qpos, qvel, h)
% then approximates
%   qpos_dot ≈ (qpos_next - qpos)/h
%
% INPUTS
%   model : MuJoCo model
%   qpos  : nq x 1
%   qvel  : nv x 1
%
% OUTPUT
%   qpos_dot : nq x 1

    h = 1e-8;   % small step for numerical derivative

    qpos_next = qpos;

    % Quaternion-aware integration in configuration space
    % Replace with your wrapper's exact function name if needed
    mj_integratePos(model, qpos_next, qvel, h);

    qpos_dot = (qpos_next - qpos) / h;
end