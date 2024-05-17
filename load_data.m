clear;
clc;

str1='orl_face/u';
str2='/';
str3='.png';
data=[];
for j=1:40
    for i=1:10
        text_string=strcat(str1,num2str(j));
        text_string=strcat(text_string,str2);
        text_string=strcat(text_string,num2str(i));
        text_string=strcat(text_string,str3);
        A=imread(text_string);
        A=imresize(A,[32 32]);
        A=double(A);
        A=reshape(A,1,length(A)*length(A));
        MappedData = mapminmax(A, 0, 1);
        MappedData = [MappedData,j];
        data=[data;MappedData];
    end
end

xlswrite('orl_data.xlsx',data)