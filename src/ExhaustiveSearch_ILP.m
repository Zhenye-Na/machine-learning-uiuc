f_1 = [0;1];
f_2 = [1;0];
f_12 = [0;0;0;0];
%f_1 = randn(2,1);
%f_2 = randn(2,1);
%f_12 = randn(4,1);

%exhaustive search:
Fmax = -inf;
y_opt = [-1 -1];
for y_1=1:2
    for y_2=1:2
        F = f_1(y_1) + f_2(y_2) + f_12(y_1 + 2*(y_2-1));
        if F>Fmax
            y_ast = [y_1 y_2];
            Fmax = F;
        end
    end
end


%ILP
%3 sum to one and 4 marginalization constraints
A = [1 1 0 0 0 0 0 0;
     0 0 1 1 0 0 0 0;
     0 0 0 0 1 1 1 1;
     -1 0 0 0 1 0 1 0;
     0 -1 0 0 0 1 0 1;
     0 0 -1 0 1 1 0 0;
     0 0 0 -1 0 0 1 1];
 rhs = [1;1;1;0;0;0;0];
 
 b = intlinprog(-[f_1;f_2;f_12], [1,2,3,4,5,6,7,8], [], [], A, rhs, zeros(8,1), ones(8,1));
 
 %Convert b to y
 [~,y_1_ilp] = max(b(1:2));
 [~,y_2_ilp] = max(b(3:4));
 F_ilp = b.'*[f_1;f_2;f_12];
 
 fprintf('Fmax = %f; F_ilp = %f\n', Fmax, F_ilp);
 fprintf('y_1 = %d; y_2 = %d; y_1_ilp = %d y_2_ilp = %d\n', y_ast(1), y_ast(2), y_1_ilp, y_2_ilp);
 
 