import { Component, OnInit } from '@angular/core';
import { ModelService } from '../model.service';
import { Model } from '../model';
import {Router, ActivatedRoute, Params} from '@angular/router';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.less']
})
export class HomeComponent implements OnInit {
  models: Model[];
  info: string;

  constructor(private modelService : ModelService, private activatedRoute : ActivatedRoute) { }

  ngOnInit() {
    this.getModels();
  }

  getModels(): void {
    this.modelService.getModels()
      .subscribe(models => this.models = models);

    this.activatedRoute.queryParams.subscribe(params => {
      if (params['savedModel'])
      {
        this.info = "Saved model " + params['savedModel'] + " successfully."
      }
    });
  }

}
