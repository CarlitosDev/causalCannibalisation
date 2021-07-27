import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fcn_helpers as fhelp
from causalimpact import CausalImpact

fcn_compare = lambda a,b: abs(a-b)/max(a,b)

def compare_promo_regular_sales(df_store, df_components, sku_A, idx_holiday_to_exclude, min_promo_days=3, min_regular_days=6):
    '''
        Explain this well as it will go on the paper...
        
        This method splits a CFAV SKU into promotional and regular chunks taking into account the inferred availability (using LOESS)\
        and the holiday periods (any other event can be included in that idx).
        
        The method divides the selling days into sequences of regular and normal days and calculates marginal sales.
    
        'avg_promo_sales' and 'avg_regular_sales' are the sales aggregated across all the slots, whereas
        'slot_promo_avg_sales' and 'slot_promo_avg_sales' represent each slot.
        
        The total sales and total days are not returned, will I need them?
        
        min_promo_days=3, min_regular_days=6 decide the minimum number of days to be taken into consideration for the sequences.
        
            TO-DO: Add the beginning and end of the promotional periods
        
        Updates:
        25.10.2020 - First attempt
    
    '''
    category_id, dept_id, sku_id_A, store_name = fhelp.get_taxonomy_from_sku_name_CFAV(sku_A)
    state_name = store_name.split('_')[0]

    sales_sku_A = df_store[f'sales-{sku_id_A}-{store_name}']
    promo_sku_A = df_store[f'promotion_flag-{sku_id_A}-{store_name}']
    inferred_availability_sku_A = df_components[f'heartbeat_flag-{sku_id_A}-{store_name}']
    
    analysis_results = []
    
    # only if there are promos
    if promo_sku_A.sum() > 0:

        availability_sku_A = inferred_availability_sku_A & (~idx_holiday_to_exclude)
        availability_value_sku_A = availability_sku_A.sum()/len(availability_sku_A)

        # Split the promotions into slots
        idx_pre_intervention, idx_post_intervention = fhelp.split_promos_into_sequences(promo_sku_A, \
            min_promo_days=min_promo_days, min_regular_days=min_regular_days)

        num_promo_slots = len(idx_pre_intervention)

        slot_promo_sales = np.zeros(num_promo_slots)
        slot_regular_sales = np.zeros(num_promo_slots)

        slot_promo_days = np.zeros(num_promo_slots)
        slot_regular_days = np.zeros(num_promo_slots)

        for idx_promo_slot in range(0, num_promo_slots):
            idx_pre_intervention_current = idx_pre_intervention[idx_promo_slot]
            idx_post_intervention_current = idx_post_intervention[idx_promo_slot]

            slot_promo_sales[idx_promo_slot] = sales_sku_A[idx_post_intervention_current].sum()
            slot_promo_days[idx_promo_slot] = idx_post_intervention_current.sum()

            slot_regular_sales[idx_promo_slot] = sales_sku_A[idx_pre_intervention_current].sum()
            slot_regular_days[idx_promo_slot] = idx_pre_intervention_current.sum()

        slot_promo_avg_sales = np.divide(slot_promo_sales, slot_promo_days)
        slot_regular_avg_sales = np.divide(slot_regular_sales, slot_regular_days)
        # totals
        total_slot_promo_days = slot_promo_days.sum()
        if total_slot_promo_days>0:
            avg_promo_sales = slot_promo_sales.sum()/total_slot_promo_days
        else:
            avg_promo_sales = 0
        
        total_slot_regular_days = slot_regular_days.sum()
        if total_slot_regular_days>0:
            avg_regular_sales = slot_regular_sales.sum()/total_slot_regular_days
        else:
            avg_regular_sales = 0
            

        # difference between the averages during promo and regular
        difference_averages_promo_to_regular = avg_promo_sales-avg_regular_sales
        # cumulative difference
        cum_difference_sales_promo_to_regular = slot_promo_sales.sum()-slot_regular_sales.sum()
        
        analysis_results.append({'num_promo_slots': num_promo_slots,
        'avg_promo_sales': avg_promo_sales,
        'avg_regular_sales': avg_regular_sales,
        'promo_days': total_slot_promo_days, 
        'regular_days':total_slot_regular_days,
        'difference_averages_promo_to_regular': difference_averages_promo_to_regular,
        'cum_difference_sales_promo_to_regular': cum_difference_sales_promo_to_regular,
        'slot_promo_avg_sales': slot_promo_avg_sales,
        'slot_regular_avg_sales': slot_regular_avg_sales,
        'availability_value_sku_A': availability_value_sku_A})
    
    return analysis_results




def calculate_causal_impact(sku_id_A, promo_sku_A, availability_sku_A, sales_sku_A, \
    sku_id_B, promo_sku_B, availability_sku_B, sales_sku_B, \
    idx_pre_intervention, idx_post_intervention, \
    idx_holiday_to_exclude: 'bool pd.Series',
    min_diff_in_units_from_reg_to_promo, min_ratio_change = 0.3,
    do_exclude_promos_SKU_B = True, be_verbose=True, \
    min_overlapping_days_regular=5, min_overlapping_days_promo=3):

    # use this flag to exclude sku_B if on promo
    total_days = promo_sku_A.shape[0]
    num_days = np.arange(total_days)
    combined_availability = availability_sku_A & availability_sku_B & (~idx_holiday_to_exclude)

    causal_analysis = []
    
    total_slots = len(idx_pre_intervention)

    for idx_promo_slot in range(0, total_slots):

        idx_pre_intervention_current = idx_pre_intervention[idx_promo_slot]
        idx_post_intervention_current = idx_post_intervention[idx_promo_slot]

        # # #
        # promo days == 'post-intervention'
        # # #
        idx_overlapping_days_promo = combined_availability & idx_post_intervention_current
        total_overlapping_days_promo = idx_overlapping_days_promo.sum()


        # overlapping promo days. Both SKUs on promo, "competing promos"
        idx_competing_promo_days = idx_overlapping_days_promo & promo_sku_B
        competing_promo_days = idx_competing_promo_days.sum()


        # # #
        # regular days == 'pre-intervention'
        # # #
        # A period should not be marked as 'regular' if SKU_B is on promotion
        if do_exclude_promos_SKU_B:
            idx_overlapping_days_regular = combined_availability & idx_pre_intervention_current & (~promo_sku_B)
        else:
            idx_overlapping_days_regular = combined_availability & idx_pre_intervention_current
        total_overlapping_days_regular = idx_overlapping_days_regular.sum()
        
        # Minimum requirements of overlap. Otherwise the analysis does not make much sense.
        # numerical index (for Causal Impact)
        if (total_overlapping_days_regular>=min_overlapping_days_regular) & \
            (total_overlapping_days_promo>=min_overlapping_days_promo):
            
            ind_regular_days = num_days[idx_overlapping_days_regular]
            start_regular = ind_regular_days.min()
            end_regular = ind_regular_days.max()
            
            ind_promo_days = num_days[idx_overlapping_days_promo]
            start_promo = ind_promo_days.min()
            end_promo = ind_promo_days.max()

            # sales of SKU_B
            # during regular
            sku_B_regular_avg_sales = sales_sku_B[idx_overlapping_days_regular].mean()
            # during promo
            sku_B_avg_sales_during_promo_sku_A = sales_sku_B[idx_overlapping_days_promo].mean()
            # Difference in average sales between the regular and the promotional one.
            diff_in_units_from_reg_to_promo = sku_B_regular_avg_sales-sku_B_avg_sales_during_promo_sku_A
            
            # can we review the post promotional sales?
            post_period_start = end_promo+1
            post_period_end = np.min([end_promo+8, total_days])
            post_promo_days = post_period_end-post_period_start+1
            sku_B_regular_post_promo_avg_sales = sales_sku_B[post_period_start:post_period_end].mean()
            # post promo should be larger than during the cannibalisation
            diff_in_units_from_promo_to_pos_promo = sku_B_avg_sales_during_promo_sku_A-sku_B_regular_post_promo_avg_sales
            
            post_promo_flag = (-diff_in_units_from_promo_to_pos_promo > min_diff_in_units_from_reg_to_promo*0.25)
            
            '''
            if idx_promo_slot+1 < total_slots:
                # during regular
                idx_reg_post_intervention = idx_pre_intervention[idx_promo_slot+1]
                sku_B_regular_post_promo_avg_sales = sales_sku_B[idx_reg_post_intervention].mean()
                # post promo should be larger than during the cannibalisation
                diff_in_units_from_promo_to_pos_promo = sku_B_avg_sales_during_promo_sku_A-sku_B_regular_post_promo_avg_sales
                post_promo_flag = (-diff_in_units_from_promo_to_pos_promo>min_diff_in_units_from_reg_to_promo)
            else:
                diff_in_units_from_promo_to_pos_promo = np.nan
                post_promo_flag = True
            '''

            # delta
            ratio_change = fcn_compare(sku_B_avg_sales_during_promo_sku_A, sku_B_regular_avg_sales)
            if be_verbose:
                print(f'Summary of the current scenario (slot {idx_promo_slot})')
                print(f'Before SKU A going on promo, the SKUs overlap for {total_overlapping_days_regular} days (promos on sku_B excluded: {do_exclude_promos_SKU_B})')
                if end_regular+1<start_promo:
                    print(f'There is a gap of {start_promo-end_regular} days between the regular days and the beginning of the promotion (due to availability/promotional period of SKU_B)')
                print(f'When SKU_A is on promo, the SKUs overlap for {total_overlapping_days_promo} days')
                print(f'During the overlapping days, SKU B is on promo for {competing_promo_days} days')

                print(f'Average sales of sku B before sku A on promo {sku_B_regular_avg_sales:.2f}')
                print(f'Average sales of sku B during sku A on promo {sku_B_avg_sales_during_promo_sku_A:.2f}')
                print(f'Average sales of sku B after sku A on promo {sku_B_regular_post_promo_avg_sales:.2f} over {post_promo_days} days - {post_promo_flag}')
                print(f'Diff in units from regular to promotion {-diff_in_units_from_reg_to_promo:.2f}')
                print(f'Diff in units from cannibalisation to regular {-diff_in_units_from_promo_to_pos_promo:.2f}')
                
                print(f'Ratio of change {ratio_change:3.2f} (the lower the closer (0,1)) {ratio_change>min_ratio_change} \n')


            if (ratio_change>min_ratio_change) & (diff_in_units_from_reg_to_promo > min_diff_in_units_from_reg_to_promo) & post_promo_flag:

                idx_regular_days = np.array([start_regular, end_regular]).tolist()
                idx_promo_days   = np.array([start_promo, end_promo]).tolist()

                print('Running Causal Impact...')
                ci = CausalImpact(sales_sku_B, idx_regular_days, idx_promo_days)

                #ci.lower_upper_percentile
                avg_actual = ci.summary_data.loc['actual', 'average']
                # Had the promo not been launched on the cannibal, we would have sold this amount.
                avg_predicted = ci.summary_data.loc['predicted', 'average']
                avg_abs_effect = ci.summary_data.loc['abs_effect', 'average']
                # This can be seen as the number of units that the cannibal is taking from the victim
                cum_abs_effect = ci.summary_data.loc['abs_effect', 'cumulative']
                posterior_tail_prob = ci.p_value
                prob_causal_effect = (1-ci.p_value)*100
                print(f'CausalImpact >> Probability of a causal event {prob_causal_effect:.2f}')

                temp_dict = {'cannibal':sku_id_A,
                            'victim':sku_id_B,
                            'slot_number': idx_promo_slot,
                            'idx_regular_days': idx_regular_days,
                            'idx_promo_days': idx_promo_days,
                            'total_overlapping_days_regular': total_overlapping_days_regular,
                            'regular_to_promo_gap': start_promo-end_regular-1,
                            'total_overlapping_days_promo': total_overlapping_days_promo,
                            'competing_promo_days': competing_promo_days,
                            'sku_B_regular_avg_sales': sku_B_regular_avg_sales,
                            'sku_B_avg_sales_during_promo_sku_A': sku_B_avg_sales_during_promo_sku_A,
                            'diff_in_units_from_reg_to_promo': diff_in_units_from_reg_to_promo,
                            'diff_in_units_from_promo_to_pos_promo': diff_in_units_from_promo_to_pos_promo,
                            'ratio_change': ratio_change,
                            'avg_actual':avg_actual,
                            'avg_predicted': avg_predicted,
                            'avg_abs_effect': avg_abs_effect,
                            'cum_abs_effect': cum_abs_effect,
                            'posterior_tail_prob': posterior_tail_prob,
                            'prob_causal_effect': prob_causal_effect
                            }
                causal_analysis.append(temp_dict)
            
    return causal_analysis



def calculate_causal_impact_with_covariates(sku_id_A, promo_sku_A, availability_sku_A, sales_sku_A, \
    sku_id_B, promo_sku_B, availability_sku_B, df_sales_covariates, \
    idx_pre_intervention, idx_post_intervention, \
    idx_holiday_to_exclude: 'bool pd.Series',
    min_diff_in_units_from_reg_to_promo, 
    min_ratio_change = 0.3,
    do_exclude_promos_SKU_B = True, be_verbose=True, \
    min_overlapping_days_regular=5, min_overlapping_days_promo=3):
    '''
        This is the method used to populate the paper results
    '''

    # use this flag to exclude sku_B if on promo
    total_days = promo_sku_A.shape[0]
    num_days = np.arange(total_days)
    combined_availability = availability_sku_A & availability_sku_B & (~idx_holiday_to_exclude)

    causal_analysis = []
    
    sales_sku_B = df_sales_covariates.iloc[:,0]
    
    total_slots = len(idx_pre_intervention)

    for idx_promo_slot in range(0, total_slots):

        idx_pre_intervention_current = idx_pre_intervention[idx_promo_slot]
        idx_post_intervention_current = idx_post_intervention[idx_promo_slot]

        # # #
        # promo days == 'post-intervention'
        # # #
        idx_overlapping_days_promo = combined_availability & idx_post_intervention_current
        total_overlapping_days_promo = idx_overlapping_days_promo.sum()


        # overlapping promo days. Both SKUs on promo, "competing promos"
        idx_competing_promo_days = idx_overlapping_days_promo & promo_sku_B
        competing_promo_days = idx_competing_promo_days.sum()


        # # #
        # regular days == 'pre-intervention'
        # # #
        # A period should not be marked as 'regular' if SKU_B is on promotion
        if do_exclude_promos_SKU_B:
            idx_overlapping_days_regular = combined_availability & idx_pre_intervention_current & (~promo_sku_B)
        else:
            idx_overlapping_days_regular = combined_availability & idx_pre_intervention_current
        total_overlapping_days_regular = idx_overlapping_days_regular.sum()
        
        # Minimum requirements of overlap. Otherwise the analysis does not make much sense.
        # numerical index (for Causal Impact)
        if (total_overlapping_days_regular>=min_overlapping_days_regular) & \
            (total_overlapping_days_promo>=min_overlapping_days_promo):
            
            ind_regular_days = num_days[idx_overlapping_days_regular]
            start_regular = ind_regular_days.min()
            end_regular = ind_regular_days.max()
            
            ind_promo_days = num_days[idx_overlapping_days_promo]
            start_promo = ind_promo_days.min()
            end_promo = ind_promo_days.max()

            # sales of SKU_B
            # during regular
            sku_B_regular_avg_sales = sales_sku_B[idx_overlapping_days_regular].mean()
            # during promo
            sku_B_avg_sales_during_promo_sku_A = sales_sku_B[idx_overlapping_days_promo].mean()
            # Difference in average sales between the regular and the promotional one.
            diff_in_units_from_reg_to_promo = sku_B_regular_avg_sales-sku_B_avg_sales_during_promo_sku_A
            
            # can we review the post promotional sales?
            post_period_start = end_promo+1
            post_period_end = np.min([end_promo+8, total_days])
            post_promo_days = post_period_end-post_period_start+1
            sku_B_regular_post_promo_avg_sales = sales_sku_B[post_period_start:post_period_end].mean()
            # post promo should be larger than during the cannibalisation
            diff_in_units_from_promo_to_pos_promo = sku_B_avg_sales_during_promo_sku_A-sku_B_regular_post_promo_avg_sales
            
            post_promo_flag = (-diff_in_units_from_promo_to_pos_promo > min_diff_in_units_from_reg_to_promo*0.25)
            
            '''
            if idx_promo_slot+1 < total_slots:
                # during regular
                idx_reg_post_intervention = idx_pre_intervention[idx_promo_slot+1]
                sku_B_regular_post_promo_avg_sales = sales_sku_B[idx_reg_post_intervention].mean()
                # post promo should be larger than during the cannibalisation
                diff_in_units_from_promo_to_pos_promo = sku_B_avg_sales_during_promo_sku_A-sku_B_regular_post_promo_avg_sales
                post_promo_flag = (-diff_in_units_from_promo_to_pos_promo>min_diff_in_units_from_reg_to_promo)
            else:
                diff_in_units_from_promo_to_pos_promo = np.nan
                post_promo_flag = True
            '''

            # delta
            ratio_change = fcn_compare(sku_B_avg_sales_during_promo_sku_A, sku_B_regular_avg_sales)
            if be_verbose:
                print(f'Summary of the current scenario (slot {idx_promo_slot})')
                print(f'Before SKU A going on promo, the SKUs overlap for {total_overlapping_days_regular} days (promos on sku_B excluded: {do_exclude_promos_SKU_B})')
                if end_regular+1<start_promo:
                    print(f'There is a gap of {start_promo-end_regular} days between the regular days and the beginning of the promotion (due to availability/promotional period of SKU_B)')
                print(f'When SKU_A is on promo, the SKUs overlap for {total_overlapping_days_promo} days')
                print(f'During the overlapping days, SKU B is on promo for {competing_promo_days} days')

                print(f'Average sales of sku B before sku A on promo {sku_B_regular_avg_sales:.2f}')
                print(f'Average sales of sku B during sku A on promo {sku_B_avg_sales_during_promo_sku_A:.2f}')
                print(f'Average sales of sku B after sku A on promo {sku_B_regular_post_promo_avg_sales:.2f} over {post_promo_days} days - {post_promo_flag}')
                print(f'Diff in units from regular to promotion {-diff_in_units_from_reg_to_promo:.2f}')
                print(f'Diff in units from cannibalisation to regular {-diff_in_units_from_promo_to_pos_promo:.2f}')
                
                print(f'Ratio of change {ratio_change:3.2f} (the lower the closer (0,1)) {ratio_change>min_ratio_change} \n')


            if (ratio_change>min_ratio_change) & (diff_in_units_from_reg_to_promo > min_diff_in_units_from_reg_to_promo) & post_promo_flag:

                idx_regular_days = np.array([start_regular, end_regular]).tolist()
                idx_promo_days   = np.array([start_promo, end_promo]).tolist()

                print('Running Causal Impact...')
                ci = CausalImpact(df_sales_covariates, idx_regular_days, idx_promo_days)
                '''
                https://github.com/dafiti/causalimpact/blob/8d881fc5c270348d8c8ff59c936997a75d7c5fac/causalimpact/main.py#L88
                
                First column must contain the `y` measured value 
                while the others contain the covariates `X` that are used in the 
                linear regression component of the model.
                '''

                #ci.lower_upper_percentile
                avg_actual = ci.summary_data.loc['actual', 'average']
                # Had the promo not been launched on the cannibal, we would have sold this amount.
                avg_predicted = ci.summary_data.loc['predicted', 'average']
                avg_abs_effect = ci.summary_data.loc['abs_effect', 'average']
                # This can be seen as the number of units that the cannibal is taking from the victim
                cum_abs_effect = ci.summary_data.loc['abs_effect', 'cumulative']
                posterior_tail_prob = ci.p_value
                prob_causal_effect = (1-ci.p_value)*100
                print(f'CausalImpact >> Probability of a causal event {prob_causal_effect:.2f}')

                temp_dict = {'cannibal':sku_id_A,
                            'victim':sku_id_B,
                            'slot_number': idx_promo_slot,
                            'idx_regular_days': idx_regular_days,
                            'idx_promo_days': idx_promo_days,
                            'total_overlapping_days_regular': total_overlapping_days_regular,
                            'regular_to_promo_gap': start_promo-end_regular-1,
                            'total_overlapping_days_promo': total_overlapping_days_promo,
                            'competing_promo_days': competing_promo_days,
                            'sku_B_regular_avg_sales': sku_B_regular_avg_sales,
                            'sku_B_avg_sales_during_promo_sku_A': sku_B_avg_sales_during_promo_sku_A,
                            'diff_in_units_from_reg_to_promo': diff_in_units_from_reg_to_promo,
                            'diff_in_units_from_promo_to_pos_promo': diff_in_units_from_promo_to_pos_promo,
                            'ratio_change': ratio_change,
                            'avg_actual':avg_actual,
                            'avg_predicted': avg_predicted,
                            'avg_abs_effect': avg_abs_effect,
                            'cum_abs_effect': cum_abs_effect,
                            'posterior_tail_prob': posterior_tail_prob,
                            'prob_causal_effect': prob_causal_effect
                            }
                causal_analysis.append(temp_dict)
            
    return causal_analysis