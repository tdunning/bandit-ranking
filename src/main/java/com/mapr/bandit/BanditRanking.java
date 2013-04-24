package com.mapr.bandit;

import com.google.common.collect.Lists;
import com.mapr.stats.bandit.BanditFactory;
import com.mapr.stats.bandit.BayesianBandit;
import com.mapr.stats.bandit.BetaBayesFactory;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.RandomWrapper;
import org.apache.mahout.math.jet.random.Uniform;

import java.util.Arrays;
import java.util.List;

/**
 * Implements a Bandit ranking.
 */
public class BanditRanking {
    private static RandomWrapper gen = RandomUtils.getRandom();

    public static void main(String[] args) {
        int keyItems = 10, pageSize = 20, totalItems = 200;

        if (args.length > 0) {
            keyItems = Integer.parseInt(args[0]);
        }

        if (args.length > 1) {
            pageSize = Integer.parseInt(args[1]);
        }

        if (args.length > 2) {
            totalItems = Integer.parseInt(args[2]);
        }

        BanditFactory bf = new BetaBayesFactory();
        List<BayesianBandit> bandit = Lists.newArrayList();
        for (int m = 0; m < 50; m++) {
            bandit.add(bf.createBandit(totalItems, gen));
        }

        double[] prob = new double[totalItems];
        Uniform u = new Uniform(gen);
        for (int j = 0; j < totalItems; j++) {
            prob[j] = u.nextDouble();
        }
        Arrays.sort(prob);

        for (int i = 0; i < 1000; i++) {

            double precision = 0;
            for (int m = 0; m < 50; m++) {
                List<Integer> page = bandit.get(m).rank(pageSize);
                for (Integer item : page) {
                    if (item < keyItems) {
                        precision++;
                    }
                }
                for (int j = 0; j < pageSize; j++) {
                    int k = page.get(j);
                    bandit.get(m).train(k, u.nextDouble() > prob[k] ? 0 : 1);
                }
            }
            precision /= keyItems * 50.0;

            System.out.printf("%d,%.1f\n", i, precision * 100);
        }
    }
}
