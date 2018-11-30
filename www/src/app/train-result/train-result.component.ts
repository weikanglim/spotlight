import { Component, OnInit } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { TrainingResult } from '../train-result';

@Component({
  selector: 'app-train-result',
  templateUrl: './train-result.component.html',
  styleUrls: ['./train-result.component.less']
})
export class TrainResultComponent implements OnInit {
  trainingResult: TrainingResult;

  constructor(private activatedRoute: ActivatedRoute) { }

  ngOnInit() {
    this.activatedRoute.queryParams.subscribe(
      params => {
        console.log(params);
        this.trainingResult = <TrainingResult> JSON.parse(params["result"]);
      }
    )
  }

}
