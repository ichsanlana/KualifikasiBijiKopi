clc; clear; close all; warning off all;

% pengujian
% arabika
% membaca file citra
nama_folder = 'data uji/arabika';
% membaca nama file yang berekstensi .jpg
nama_file = dir(fullfile(nama_folder,'*.jpg'));
% membaca jumlah file
jumlah_file = numel(nama_file);
%menginisialisasi variabel ciri arabika dan target arabika
ciri_arabika = zeros(jumlah_file,6);
target_arabika = zeros(jumlah_file,1);
% melakukan pengolahan citra terhadap seluruh file
for k = 1:jumlah_file
    %membaca file citra rgb
    img = imread(fullfile(nama_folder,nama_file(k).name));
    %mengkonversi citra rgb menjadi citra grayscale
    img_gray = rgb2gray(img);
    %melakukan ekstraksi ciri tekstur orde satu
    %membaca ukuran citra
    [m,n] = size(img_gray);
    % menghitung frekuensi aras keabuan
    L = 256;
    frek = zeros(L,1);
    F = double(img_gray);
    for i = 1:m
        for j =1:n
        intensitas = F(i,j);
        frek(intensitas+1) = frek(intensitas+1)+1;
        end
    end
    %menghitung probabilitas
    jum_piksel = m*n;
    for i = 0:L-1
        Prob(i+1)= frek(i+1)/jum_piksel;
    end
    % menghitung nilai mu
    mu = 0;
    for i = 1:L-1
        mu = mu+i*Prob(i+1);
    end
    % menghitung nilai standar deviasi
    varians = 0;
    for i = 0:L-1
        varians = varians + (i-mu)^2*Prob(i+1);
    end
    deviasi = sqrt(varians);
    varians_n = varians/(L-1)*2;  %proses normalisasi
    % menghitung nilai skewness
    skewness = 0;
    for i = 0:L-1
        skewness = skewness+(i-mu)^3*Prob(i+1);
    end
    skewness = skewness/(L-1)^2;  %normalisasi
    %menghitung nilai energi
    energi = 0;
    for i = 0:L-1
        energi = energi+Prob(i+1)^2;
    end
    % menghitung nilai entropi
    entropi = 0;
    for i = 0:L-1
        if Prob(i+1) ~=0
            entropi = entropi+Prob(i+1)*log(Prob(i+1));
        end
    end
    entropi = -entropi;
    % menghitung nilai smoothness
    smoothness = 1-1/(1+varians_n);
    %mengisi variabel ciri_arabika dengan fitur hasil ekstraksi
    ciri_arabika(k,1) = mu;
    ciri_arabika(k,2) = deviasi;
    ciri_arabika(k,3) = skewness;
    ciri_arabika(k,4) = energi;
    ciri_arabika(k,5) = entropi;
    ciri_arabika(k,6) = smoothness;
    %mengisi variabel target_arabika dengan angka 1
    target_arabika(k) = 1;
end
    

% robusta
% membaca file citra
nama_folder = 'data uji/robusta';
% membaca nama file yang berekstensi .jpg
nama_file = dir(fullfile(nama_folder,'*.jpg'));
% membaca jumlah file
jumlah_file = numel(nama_file);
%menginisialisasi variabel ciri robusta dan target robusta
ciri_robusta = zeros(jumlah_file,6);
target_robusta = zeros(jumlah_file,1);
% melakukan pengolahan citra terhadap seluruh file
for k = 1:jumlah_file
    %membaca file citra rgb
    img = imread(fullfile(nama_folder,nama_file(k).name));
    %mengkonversi citra rgb menjadi citra grayscale
    img_gray = rgb2gray(img);
    %melakukan ekstraksi ciri tekstur orde satu
    %membaca ukuran citra
    [m,n] = size(img_gray);
    % menghitung frekuensi aras keabuan
    L = 256;
    frek = zeros(L,1);
    F = double(img_gray);
    for i = 1:m
        for j =1:n
        intensitas = F(i,j);
        frek(intensitas+1) = frek(intensitas+1)+1;
        end
    end
    %menghitung probabilitas
    jum_piksel = m*n;
    for i = 0:L-1
        Prob(i+1)= frek(i+1)/jum_piksel;
    end
    % menghitung nilai mu
    mu = 0;
    for i = 1:L-1
        mu = mu+i*Prob(i+1);
    end
    % menghitung nilai standar deviasi
    varians = 0;
    for i = 0:L-1
        varians = varians + (i-mu)^2*Prob(i+1);
    end
    deviasi = sqrt(varians);
    varians_n = varians/(L-1)*2;  %proses normalisasi
    % menghitung nilai skewness
    skewness = 0;
    for i = 0:L-1
        skewness = skewness+(i-mu)^3*Prob(i+1);
    end
    skewness = skewness/(L-1)^2;  %normalisasi
    %menghitung nilai energi
    energi = 0;
    for i = 0:L-1
        energi = energi+Prob(i+1)^2;
    end
    % menghitung nilai entropi
    entropi = 0;
    for i = 0:L-1
        if Prob(i+1) ~=0
            entropi = entropi+Prob(i+1)*log(Prob(i+1));
        end
    end
    entropi = -entropi;
    % menghitung nilai smoothness
    smoothness = 1-1/(1+varians_n);
    %mengisi variabel ciri_robusta dengan fitur hasil ekstraksi
    ciri_robusta(k,1) = mu;
    ciri_robusta(k,2) = deviasi;
    ciri_robusta(k,3) = skewness;
    ciri_robusta(k,4) = energi;
    ciri_robusta(k,5) = entropi;
    ciri_robusta(k,6) = smoothness;
    %mengisi variabel target_robusta dengan angka 2
    target_robusta(k) = 2;
end

% menyusun variabel ciri_uji dan target_uji
ciri_uji = [ciri_arabika;ciri_robusta];
target_uji = [target_arabika;target_robusta];

% melakukan operasi transpose terhadap variabel ciri_uji dan target_uji

X = ciri_uji';
T = ind2vec(target_uji');

%memanggil arsitektur hasil pelatihan
load jaringan

%membaca nilai keluaran hasil peujian
Y = sim(jaringan,X);
hasil_uji = vec2ind(Y);

%menghitung akurasi pengujian
jumlah_benar = 0;
jumlah_data = numel(hasil_uji);
for k = 1:jumlah_data
    if isequal(hasil_uji(k),target_uji(k))
        jumlah_benar = jumlah_benar+1;
    end
end
akurasi_pengujian = jumlah_benar/jumlah_data*100




