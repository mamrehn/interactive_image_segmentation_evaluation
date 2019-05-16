"use strict";

const app = angular.module('EFKT', ['ngMaterial', 'ngAnimate']);

app.controller('SliderCtrl', function($scope) {
    $scope.windowing_slider = {
        width_val: 255,
        center_val: 127
    };
    $scope.contour_slider = {
        value: 7
    };

    $scope.$watch('windowing_slider', function (newValue, oldValue) {
        window.dispatchEvent(new StorageEvent('windowing_slider_event', {
                key: 'windowing_slider',
                newValue: newValue['width_val'] + ',' + (newValue['center_val'])
            })
        );
    }, true);

    $scope.$watch('contour_slider', function (newValue, oldValue) {
        window.dispatchEvent(new StorageEvent('contour_slider_event', {
                key: 'contour_slider',
                newValue: newValue['value']
            })
        );
    }, true);
});

app.config(function($mdThemingProvider) {
    const white_pal = $mdThemingProvider.extendPalette('red', {
        '500': '#cccccc',
        'contrastDefaultColor': 'dark'
    });
    // Register the new color palette map with the name white_pal
    $mdThemingProvider.definePalette('white_pal', white_pal);
    // Use that theme for the primary intentions
    $mdThemingProvider.theme('default').primaryPalette('white_pal');
});
