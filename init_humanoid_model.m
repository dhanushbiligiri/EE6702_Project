function [model, data, info] = init_humanoid_model(xml_path)
% Initialize MuJoCo humanoid model and data
%
% INPUT
%   xml_path : full path to humanoid.xml
%
% OUTPUT
%   model, data : MuJoCo model/data handles
%   info        : struct with dimensions

    % Example:
    % xml_path = '/path/to/humanoid.xml';

    % Load model from XML
    % Depending on your MATLAB MuJoCo wrapper, this may be:
    %   model = mj_loadXML(xml_path, '');
    % or:
    %   model = mujoco.mj_loadXML(xml_path, []);
    %
    % Replace the next line with the exact constructor name used by your binding.
    model = mj_loadXML(xml_path);

    % Allocate data
    data = mj_makeData(model);

    % Basic dimensions
    info.nq = model.nq;   % configuration dimension
    info.nv = model.nv;   % velocity dimension
    info.nu = model.nu;   % control dimension

    fprintf('Loaded model: %s\n', xml_path);
    fprintf('nq = %d, nv = %d, nu = %d\n', info.nq, info.nv, info.nu);
end