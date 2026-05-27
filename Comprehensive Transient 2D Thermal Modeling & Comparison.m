% part1-Arshia Massoompour-404668189
% Comprehensive Transient 2D Thermal Modeling & Comparison

clear; clc; close all;

% 1. EXPLICIT METHOD 
k = 15; 
rho = 3600;
Cp = 900;
alpha = k / (rho * Cp);
R = 0.005; L = 0.02;
Nr = 20; 
Nz = 45; % 20 radial divisions, 45 axial divisions
dr = R/(Nr-1);
dz = L/(Nz-1); 
r = linspace(0, R, Nr);
z = linspace(0, L, Nz);

% Stability Condition 
dt = (min(dr, dz)^2) / (4 * alpha); 
t_final = 20; 
steps = ceil(t_final/dt);
T = 25 * ones(Nr, Nz); % Initial condition
T_center_exp = zeros(steps, 1);

tic;
for n = 1:steps
    T_old = T;
    T_old(Nr, :) = 100; % Wall Boundary
    T_old(:, 1) = 100;  % Base Boundary
    T_old(:, Nz) = 100; % Top Boundary
    
    for i = 1:Nr-1
        for j = 2:Nz-1
            if i == 1 % Centerline (r=0) - L'Hopital's Rule
                lap = 2*(T_old(i+1,j)-T_old(i,j))/dr^2 + (T_old(i,j+1)-2*T_old(i,j)+T_old(i,j-1))/dz^2;
            else
                lap = (T_old(i+1,j)-2*T_old(i,j)+T_old(i-1,j))/dr^2 + ...
                      (1/r(i))*(T_old(i+1,j)-T_old(i-1,j))/(2*dr) + ...
                      (T_old(i,j+1)-2*T_old(i,j)+T_old(i,j-1))/dz^2;
            end
            T(i,j) = T_old(i,j) + alpha * dt * lap;
        end
    end
    T_center_exp(n) = T(1, round(Nz/2));
end
runtime_exp = toc;
save('explicit_res.mat', 'T_center_exp', 'runtime_exp');
fprintf('Explicit Method Done. Execution Time: %.4f s\n', runtime_exp);

%% =========================================================================
% 2. IMPLICIT METHOD
Nr = 15;   
Nz = 30;  
dr = R/(Nr-1);
dz = L/(Nz-1);
r = linspace(0, R, Nr);
z = linspace(0, L, Nz);
dt = 1; 
steps = ceil(t_final/dt);
N_total = Nr * Nz;
T_vec = 25 * ones(N_total, 1); 
A = eye(N_total); 
Sr = alpha * dt / dr^2;
Sz = alpha * dt / dz^2;

tic;
for j = 2:Nz-1 
    for i = 1:Nr-1 
        row = (j-1)*Nr + i; 
        
        if i == 1 
            A(row, row) = 1 + 4*Sr + 2*Sz;
            A(row, row+1) = -4*Sr;      
            A(row, row+Nr) = -Sz;       
            A(row, row-Nr) = -Sz;       
        else 
            A(row, row) = 1 + 2*Sr + 2*Sz;
            A(row, row+1) = -Sr - (alpha*dt)/(2*r(i)*dr); 
            A(row, row-1) = -Sr + (alpha*dt)/(2*r(i)*dr); 
            A(row, row+Nr) = -Sz; 
            A(row, row-Nr) = -Sz; 
        end
    end
end

for j = 1:Nz
    for i = 1:Nr
        row = (j-1)*Nr + i;
        if i == Nr || j == 1 || j == Nz
            T_vec(row) = 100; 
        end
    end
end

T_center_imp = zeros(steps, 1);
for n = 1:steps
    T_vec = A \ T_vec; 
    center_idx = round(Nz/2)*Nr + 1;
    T_center_imp(n) = T_vec(center_idx);
end
runtime_imp = toc;
save('implicit_res.mat', 'T_center_imp', 'runtime_imp', 'dt', 't_final');
fprintf('Implicit Method Done. Execution Time: %.4f s\n', runtime_imp);

%% =========================================================================
% 3. CRANK-NICOLSON METHOD 
dt = 1.0; 
steps = ceil(t_final/dt);
T_vec = 25 * ones(N_total, 1);
ML = eye(N_total); 
MR = eye(N_total); 
Sr = alpha * dt / dr^2;
Sz = alpha * dt / dz^2;

tic;
for j = 2:Nz-1
    for i = 1:Nr-1
        row = (j-1)*Nr + i;
        
        if i == 1 
            ML(row, row) = 1 + 2*Sr + Sz; 
            ML(row, row+1) = -2*Sr;
            ML(row, row+Nr) = -Sz/2;
            ML(row, row-Nr) = -Sz/2;
                        
            MR(row, row) = 1 - 2*Sr - Sz;
            MR(row, row+1) = 2*Sr;
            MR(row, row+Nr) = Sz/2;
            MR(row, row-Nr) = Sz/2;
        else 
            rad_term = (alpha*dt)/(4*r(i)*dr);
            
            ML(row, row) = 1 + Sr + Sz;
            ML(row, row+1) = -Sr/2 - rad_term;
            ML(row, row-1) = -Sr/2 + rad_term;
            ML(row, row+Nr) = -Sz/2;
            ML(row, row-Nr) = -Sz/2;
           
            MR(row, row) = 1 - Sr - Sz;
            MR(row, row+1) = Sr/2 + rad_term;
            MR(row, row-1) = Sr/2 - rad_term;
            MR(row, row+Nr) = Sz/2;
            MR(row, row-Nr) = Sz/2;
        end
    end
end

for j = 1:Nz
    for i = 1:Nr
        row = (j-1)*Nr + i;
        if i == Nr || j == 1 || j == Nz
            ML(row, :) = 0; ML(row, row) = 1;
            MR(row, :) = 0; MR(row, row) = 1;
            T_vec(row) = 100;
        end
    end
end

T_center_cn = zeros(steps, 1);
for n = 1:steps
    B = MR * T_vec;
    T_vec = ML \ B;
    
    center_idx = round(Nz/2)*Nr + 1;
    T_center_cn(n) = T_vec(center_idx);
end
runtime_cn = toc;
save('cn_res.mat', 'T_center_cn', 'runtime_cn', 'dt');
fprintf('Crank-Nicolson Done. Execution Time: %.4f s\n', runtime_cn);

%% =========================================================================
% 4. COMPARISON SCRIPT (FIGURE 1)
load('explicit_res.mat');
load('implicit_res.mat');
load('cn_res.mat');

t_exp = linspace(0, t_final, length(T_center_exp)); 
t_imp = linspace(0, t_final, length(T_center_imp));  
t_cn  = linspace(0, t_final, length(T_center_cn));

figure('Color', 'w', 'Name', 'Temporal Scheme Comparison');
hold on; grid on;
plot(t_exp, T_center_exp, 'r-', 'LineWidth', 2, 'DisplayName', 'Explicit Method');
plot(t_imp, T_center_imp, 'b--', 'LineWidth', 2, 'DisplayName', 'Implicit Method');
plot(t_cn, T_center_cn, 'g:', 'LineWidth', 2.5, 'DisplayName', 'Crank-Nicolson Method');
xlabel('Time (seconds)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Center Temperature (^{\circ}C)', 'FontSize', 12, 'FontWeight', 'bold');
title('Transient Centerline Temperature Comparison', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'southeast', 'FontSize', 10);
