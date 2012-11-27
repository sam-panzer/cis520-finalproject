%testerror = zeros(1,10);
%for depth = 1 : 10
    %model.audio_forest_test = Stochastic_Bosque(Xt_audio(1:7000,:),Yt(1:7000,:),'ntrees',25,'depth',depth);
%     P = zeros(2000,10);
%     for i=7001:9000
%         P(i-7000,:) = make_final_prediction(model,train(i));
%     end
%     YP = P(:,1);
%     testerror = sum(YP == Yt(7001:9000)) / 2000;
%end

P = zeros(9704,10);
for i=1:9704
    P(i,:) = make_final_prediction(model,train(i));
end
YP = P(:,1);
testerror = sum(YP == Yt) / 9704;