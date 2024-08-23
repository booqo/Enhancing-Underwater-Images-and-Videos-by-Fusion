function lab = rgb_to_lab(rgb)

cform = makecform('srgb2lab');  %rgb转lab公式
lab = applycform(rgb,cform);    %lab格式

test1 = rgb(:, :, 1);
Max = max(test1);
Min = min(test1);
test2 = rgb(:, :, 2);
test3 = rgb(:, :, 3);



test4 = lab(:, :, 1);
test5 = lab(:, :, 2);
test6 = lab(:, :, 3);

Max = max(test4(:));
Min = min(test4(:));

end